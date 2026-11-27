"""
ghost/client.py — Direct embeddable Python clients (sync ``Ghost`` and async ``AsyncGhost``).

Allows any Python project to call Ghost providers (ChatGPT, Gemini, Gemini ToT,
Spectre) in-process with zero server, zero port, and zero FastAPI dependency.
"""
from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any, AsyncGenerator, Callable, Iterator, TypeVar

from ghost.core.exceptions import ModelNotFoundError
from ghost.engine import GhostEngine, parse_openai_sse_chunk
from ghost.providers import ProviderRegistry

T = TypeVar("T")

_STREAM_END = object()
_HIGH_WATER_MARK = 64
_LOW_WATER_MARK = 32


class _ErrorEnvelope:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc


class _SyncStream(Iterator[T]):
    """Synchronous iterator bridging an async generator running on an ``_AsyncLoopRunner``.

    Supports ``for item in stream:``, ``with stream:``, and explicit ``stream.close()``.
    Breaking out early or closing cleanly cancels and ``aclose()``s the upstream
    async generator without blocking the event loop.
    """

    def __init__(
        self,
        runner: _AsyncLoopRunner,
        agen_factory: Callable[[], AsyncGenerator[T, None]],
    ):
        self._runner = runner
        self._queue: queue.Queue[Any] = queue.Queue()
        self._stop_event = threading.Event()
        self._done_event = threading.Event()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._can_produce: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        self._start_pump(agen_factory)

    def _start_pump(self, agen_factory: Callable[[], AsyncGenerator[T, None]]) -> None:
        loop = self._runner.ensure_loop()
        self._loop = loop

        async def _pump() -> None:
            can_produce = asyncio.Event()
            can_produce.set()
            self._can_produce = can_produce

            agen: AsyncGenerator[T, None] | None = None
            try:
                if self._stop_event.is_set():
                    return
                agen = agen_factory()
                async for item in agen:
                    while (
                        self._queue.qsize() >= _HIGH_WATER_MARK
                        and not self._stop_event.is_set()
                    ):
                        can_produce.clear()
                        await can_produce.wait()
                    if self._stop_event.is_set():
                        break
                    self._queue.put_nowait(item)
            except BaseException as exc:
                if not self._stop_event.is_set():
                    self._queue.put_nowait(_ErrorEnvelope(exc))
            finally:
                if agen is not None:
                    try:
                        await agen.aclose()
                    except Exception:
                        pass
                self._queue.put_nowait(_STREAM_END)

        def _schedule() -> None:
            task = loop.create_task(_pump())
            self._task = task
            task.add_done_callback(lambda _: self._done_event.set())
            if self._stop_event.is_set() and not task.done():
                task.cancel()

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is loop:
            _schedule()
        else:
            loop.call_soon_threadsafe(_schedule)

    def _signal_produce(self) -> None:
        if self._can_produce is not None:
            self._can_produce.set()

    def _cancel_on_loop(self) -> None:
        self._signal_produce()
        if self._task is not None and not self._task.done():
            self._task.cancel()

    def __iter__(self) -> _SyncStream[T]:
        return self

    def __next__(self) -> T:
        if self._closed:
            raise StopIteration
        item = self._queue.get()
        if (
            self._loop is not None
            and self._queue.qsize() <= _LOW_WATER_MARK
            and not self._loop.is_closed()
        ):
            try:
                self._loop.call_soon_threadsafe(self._signal_produce)
            except RuntimeError:
                pass
        if item is _STREAM_END:
            self._closed = True
            raise StopIteration
        if isinstance(item, _ErrorEnvelope):
            self._closed = True
            raise item.exc
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._loop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._cancel_on_loop)
            except RuntimeError:
                pass
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not self._loop:
            self._done_event.wait(timeout=2.0)

    def __enter__(self) -> _SyncStream[T]:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class _AsyncLoopRunner:
    """Dedicated daemon thread owning a persistent ``asyncio`` event loop.

    Keeps provider HTTP sessions and background refresh tasks alive across
    multiple synchronous calls without interfering with any event loop in the
    caller's thread (e.g. Jupyter, async web frameworks).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    def ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._closed:
                raise RuntimeError("Ghost client is closed.")
            if self._loop is not None and self._loop.is_running():
                return self._loop

            ready = threading.Event()
            created_loop: list[asyncio.AbstractEventLoop] = []

            def _run_loop() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                created_loop.append(loop)
                ready.set()
                try:
                    loop.run_forever()
                finally:
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()

            thread = threading.Thread(
                target=_run_loop, name="ghost-async-loop", daemon=True
            )
            thread.start()
            ready.wait()
            self._loop = created_loop[0]
            self._thread = thread
            return self._loop

    def run(self, coro) -> Any:
        loop = self.ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()

    def iter_async_gen(
        self, agen_factory: Callable[[], AsyncGenerator[T, None]]
    ) -> _SyncStream[T]:
        return _SyncStream(self, agen_factory)

    def close(self, cleanup_coro_factory: Callable[[], Any] | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None

        if loop is None or not loop.is_running():
            return

        if cleanup_coro_factory is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(cleanup_coro_factory(), loop)
                future.result(timeout=10.0)
            except Exception:
                pass

        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# Payload builders (shared between AsyncGhost and Ghost)
# ---------------------------------------------------------------------------

def _build_chat_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    stream: bool,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    max_completion_tokens: int | None,
    stop: str | list[str] | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    response_format: dict[str, Any] | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
    stream_options: dict[str, Any] | None,
    conversation_id: str | None,
    parent_message_id: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
        **extra,
    }
    optional_fields = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_completion_tokens,
        "stop": stop,
        "tools": tools,
        "tool_choice": tool_choice,
        "response_format": response_format,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stream_options": stream_options,
        "conversation_id": conversation_id,
        "parent_message_id": parent_message_id,
    }
    for key, value in optional_fields.items():
        if value is not None:
            payload[key] = value
    return payload


def _build_anthropic_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    system: str | list[dict[str, Any]] | None,
    stream: bool,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    stop_sequences: list[str] | None,
    temperature: float | None,
    top_p: float | None,
    metadata: dict[str, Any] | None,
    conversation_id: str | None,
    parent_message_id: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        **extra,
    }
    optional_fields = {
        "system": system,
        "tools": tools,
        "tool_choice": tool_choice,
        "stop_sequences": stop_sequences,
        "temperature": temperature,
        "top_p": top_p,
        "metadata": metadata,
        "conversation_id": conversation_id,
        "parent_message_id": parent_message_id,
    }
    for key, value in optional_fields.items():
        if value is not None:
            payload[key] = value
    return payload


def _extract_reply_text(completion: dict[str, Any]) -> str:
    choices = completion.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


# ---------------------------------------------------------------------------
# Async namespaces & client
# ---------------------------------------------------------------------------

class _AsyncCompletionsResource:
    def __init__(self, engine: GhostEngine):
        self._engine = engine

    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str = "auto",
        stream: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stop: str | list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        stream_options: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
        **extra: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        payload = _build_chat_payload(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stream_options=stream_options,
            conversation_id=conversation_id,
            parent_message_id=parent_message_id,
            extra=extra,
        )
        if stream:
            # Validate model & prepare plan eagerly before returning iterator
            plan = self._engine.prepare_chat_stream(payload, emit_error_chunks=False)

            async def _parsed_stream() -> AsyncGenerator[dict[str, Any], None]:
                try:
                    async for raw_chunk in plan.stream:
                        parsed = parse_openai_sse_chunk(raw_chunk)
                        if parsed is not None:
                            yield parsed
                finally:
                    await plan.stream.aclose()

            return _parsed_stream()
        return await self._engine.chat_completion(payload)


class _AsyncChatNamespace:
    def __init__(self, engine: GhostEngine):
        self.completions = _AsyncCompletionsResource(engine)


class _AsyncMessagesResource:
    def __init__(self, engine: GhostEngine):
        self._engine = engine

    async def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str = "gemini",
        max_tokens: int = 4096,
        system: str | list[dict[str, Any]] | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stop_sequences: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        metadata: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
        anthropic_beta: str = "",
        **extra: Any,
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        payload = _build_anthropic_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_p=top_p,
            metadata=metadata,
            conversation_id=conversation_id,
            parent_message_id=parent_message_id,
            extra=extra,
        )
        if stream:
            from ghost.engine import parse_anthropic_sse_chunk

            plan = self._engine.prepare_anthropic_stream(
                payload, beta_header=anthropic_beta, emit_error_chunks=False
            )

            async def _parsed_stream() -> AsyncGenerator[dict[str, Any], None]:
                try:
                    async for raw_chunk in plan.stream:
                        parsed = parse_anthropic_sse_chunk(raw_chunk)
                        if parsed is not None:
                            yield parsed
                finally:
                    await plan.stream.aclose()

            return _parsed_stream()
        return await self._engine.anthropic_message(payload, beta_header=anthropic_beta)

    async def count_tokens(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str = "gemini",
        system: str | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> dict[str, int]:
        body: dict[str, Any] = {"model": model, "messages": messages, **extra}
        if system is not None:
            body["system"] = system
        if tools is not None:
            body["tools"] = tools
        return self._engine.count_tokens(body)


class _AsyncModelsResource:
    def __init__(self, engine: GhostEngine):
        self._engine = engine

    async def list(self) -> dict[str, Any]:
        models = await self._engine.list_models()
        return {"object": "list", "data": models}

    async def retrieve(self, model_id: str) -> dict[str, Any]:
        model = await self._engine.get_model(model_id)
        if model is None:
            raise ModelNotFoundError(f"Model '{model_id}' not found")
        return model


class AsyncGhost:
    """Native ``asyncio`` embedded Ghost client (no server required).

    Example::

        async with AsyncGhost() as client:
            reply = await client.ask("Hello!", model="auto")
            resp = await client.chat.completions.create(
                model="gemini",
                messages=[{"role": "user", "content": "Hi"}],
            )
    """

    def __init__(
        self,
        *,
        default_provider: str | None = None,
        enable_fallback: bool = True,
        warm_up: bool = False,
        raw_logging: bool = False,
        registry: ProviderRegistry | None = None,
        engine: GhostEngine | None = None,
    ):
        self.engine = (
            engine
            if engine is not None
            else GhostEngine(
                registry=registry,
                default_provider=default_provider,
                enable_fallback=enable_fallback,
                raw_logging=raw_logging,
            )
        )
        self._warm_up = warm_up
        self.chat = _AsyncChatNamespace(self.engine)
        self.messages = _AsyncMessagesResource(self.engine)
        self.models = _AsyncModelsResource(self.engine)

    async def start(self) -> AsyncGhost:
        """Eagerly warm up provider sessions and start background token refresh."""
        await self.engine.start()
        return self

    async def aclose(self) -> None:
        """Stop background refresh tasks and close underlying HTTP clients."""
        await self.engine.aclose()

    async def __aenter__(self) -> AsyncGhost:
        if self._warm_up:
            await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    async def ask(
        self,
        prompt: str,
        *,
        model: str = "auto",
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience one-liner returning the assistant's reply text."""
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = await self.chat.completions.create(
            model=model, messages=messages, stream=False, **kwargs
        )
        return _extract_reply_text(resp)

    async def video_chat(
        self,
        video_bytes: bytes,
        prompt: str,
        *,
        model: str = "auto",
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        filename: str = "video",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Extract keyframes (requires ``ghost[video]``) and stream OpenAI chunk dicts."""
        plan = await self.engine.prepare_video_chat_stream(
            video_bytes,
            prompt,
            model=model,
            conversation_id=conversation_id,
            parent_message_id=parent_message_id,
            history=history,
            filename=filename,
            emit_error_chunks=False,
        )
        async for raw_chunk in plan.stream:
            parsed = parse_openai_sse_chunk(raw_chunk)
            if parsed is not None:
                yield parsed


# ---------------------------------------------------------------------------
# Synchronous namespaces & client
# ---------------------------------------------------------------------------

class _SyncCompletionsResource:
    def __init__(self, async_client: AsyncGhost, runner: _AsyncLoopRunner):
        self._async = async_client
        self._runner = runner

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str = "auto",
        stream: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stop: str | list[str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        response_format: dict[str, Any] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        stream_options: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
        **extra: Any,
    ) -> dict[str, Any] | _SyncStream[dict[str, Any]]:
        kwargs = dict(
            messages=messages,
            model=model,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            stream_options=stream_options,
            conversation_id=conversation_id,
            parent_message_id=parent_message_id,
            **extra,
        )
        if not stream:
            return self._runner.run(self._async.chat.completions.create(**kwargs))

        # Validate model synchronously before returning stream iterator
        agen = self._runner.run(self._async.chat.completions.create(**kwargs))
        return self._runner.iter_async_gen(lambda: agen)


class _SyncChatNamespace:
    def __init__(self, async_client: AsyncGhost, runner: _AsyncLoopRunner):
        self.completions = _SyncCompletionsResource(async_client, runner)


class _SyncMessagesResource:
    def __init__(self, async_client: AsyncGhost, runner: _AsyncLoopRunner):
        self._async = async_client
        self._runner = runner

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str = "gemini",
        max_tokens: int = 4096,
        system: str | list[dict[str, Any]] | None = None,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stop_sequences: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        metadata: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
        anthropic_beta: str = "",
        **extra: Any,
    ) -> dict[str, Any] | _SyncStream[dict[str, Any]]:
        kwargs = dict(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            system=system,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_p=top_p,
            metadata=metadata,
            conversation_id=conversation_id,
            parent_message_id=parent_message_id,
            anthropic_beta=anthropic_beta,
            **extra,
        )
        if not stream:
            return self._runner.run(self._async.messages.create(**kwargs))

        agen = self._runner.run(self._async.messages.create(**kwargs))
        return self._runner.iter_async_gen(lambda: agen)

    def count_tokens(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str = "gemini",
        system: str | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> dict[str, int]:
        return self._runner.run(
            self._async.messages.count_tokens(
                messages=messages, model=model, system=system, tools=tools, **extra
            )
        )


class _SyncModelsResource:
    def __init__(self, async_client: AsyncGhost, runner: _AsyncLoopRunner):
        self._async = async_client
        self._runner = runner

    def list(self) -> dict[str, Any]:
        return self._runner.run(self._async.models.list())

    def retrieve(self, model_id: str) -> dict[str, Any]:
        return self._runner.run(self._async.models.retrieve(model_id))


class Ghost:
    """Synchronous embedded Ghost client (no server required).

    Owns a private background event-loop thread so provider sessions stay warm
    across calls and sync code never conflicts with an existing event loop.

    Example::

        with Ghost() as client:
            print(client.ask("What is 2+2?", model="auto"))
    """

    def __init__(
        self,
        *,
        default_provider: str | None = None,
        enable_fallback: bool = True,
        warm_up: bool = False,
        raw_logging: bool = False,
        registry: ProviderRegistry | None = None,
        engine: GhostEngine | None = None,
    ):
        self._runner = _AsyncLoopRunner()
        self._async = AsyncGhost(
            default_provider=default_provider,
            enable_fallback=enable_fallback,
            warm_up=warm_up,
            raw_logging=raw_logging,
            registry=registry,
            engine=engine,
        )
        self._warm_up = warm_up
        self.chat = _SyncChatNamespace(self._async, self._runner)
        self.messages = _SyncMessagesResource(self._async, self._runner)
        self.models = _SyncModelsResource(self._async, self._runner)

    @property
    def engine(self) -> GhostEngine:
        return self._async.engine

    def start(self) -> Ghost:
        """Eagerly warm up provider sessions and start background token refresh."""
        self._runner.run(self._async.start())
        return self

    def close(self) -> None:
        """Stop background refresh tasks, close HTTP sessions, and stop the loop thread."""
        self._runner.close(cleanup_coro_factory=self._async.aclose)

    def __enter__(self) -> Ghost:
        if self._warm_up:
            self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def ask(
        self,
        prompt: str,
        *,
        model: str = "auto",
        system: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Convenience one-liner returning the assistant's reply text."""
        return self._runner.run(
            self._async.ask(prompt, model=model, system=system, **kwargs)
        )

    def video_chat(
        self,
        video_bytes: bytes,
        prompt: str,
        *,
        model: str = "auto",
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
        history: list[dict[str, Any]] | None = None,
        filename: str = "video",
    ) -> _SyncStream[dict[str, Any]]:
        """Extract keyframes (requires ``ghost[video]``) and yield OpenAI chunk dicts."""
        plan = self._runner.run(
            self._async.engine.prepare_video_chat_stream(
                video_bytes,
                prompt,
                model=model,
                conversation_id=conversation_id,
                parent_message_id=parent_message_id,
                history=history,
                filename=filename,
                emit_error_chunks=False,
            )
        )

        async def _parsed_stream() -> AsyncGenerator[dict[str, Any], None]:
            async for raw_chunk in plan.stream:
                parsed = parse_openai_sse_chunk(raw_chunk)
                if parsed is not None:
                    yield parsed

        return self._runner.iter_async_gen(_parsed_stream)


__all__ = ["Ghost", "AsyncGhost"]
