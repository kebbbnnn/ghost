"""
ghost/adapters/httpx_transport.py — Pure-httpx in-process transports & SDK factories.

Routes ``httpx.Client`` / ``httpx.AsyncClient`` requests directly into
``GhostEngine`` in memory — zero FastAPI, zero ASGI, zero TCP sockets.

This makes Ghost a true zero-server drop-in for:
  - ``openai.OpenAI`` / ``openai.AsyncOpenAI``
  - ``anthropic.Anthropic`` / ``anthropic.AsyncAnthropic``
  - Any framework accepting an ``httpx`` client or transport (LangChain, Instructor, PydanticAI, etc.)
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, AsyncIterator, Iterator

import httpx
from pydantic import ValidationError

from ghost.client import _AsyncLoopRunner, _SyncStream
from ghost.core.exceptions import ModelNotFoundError, RateLimitError
from ghost.engine import GhostEngine
from ghost.providers import ProviderRegistry

DEFAULT_OPENAI_BASE_URL = "http://ghost.local/v1"
DEFAULT_ANTHROPIC_BASE_URL = "http://ghost.local"


# ---------------------------------------------------------------------------
# Protocol header & error response builders (pure dicts, zero FastAPI deps)
# ---------------------------------------------------------------------------

def _openai_headers(
    request_id: str,
    *,
    elapsed_ms: int = 0,
    conversation_id: str | None = None,
    parent_message_id: str | None = None,
    is_stream: bool = False,
) -> dict[str, str]:
    headers = {
        "X-Request-ID": f"req_{request_id.replace('.', '')}",
        "OpenAI-Organization": "org-ghost",
        "OpenAI-Processing-Ms": str(max(1, elapsed_ms)),
        "OpenAI-Version": "2020-10-01",
    }
    if is_stream:
        headers.update(
            {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
            }
        )
    else:
        headers["Content-Type"] = "application/json"
    if conversation_id:
        headers["X-Ghost-Conversation-Id"] = conversation_id
    if parent_message_id:
        headers["X-Ghost-Parent-Message-Id"] = parent_message_id
    return headers


def _anthropic_headers(
    request_id: str,
    *,
    conversation_id: str | None = None,
    parent_message_id: str | None = None,
    is_stream: bool = False,
    anthropic_beta: str | None = None,
) -> dict[str, str]:
    headers = {
        "Request-ID": f"req_{request_id.replace('.', '')}",
        "Anthropic-Version": "2023-06-01",
        "Anthropic-Ratelimit-Requests-Limit": "1000",
        "Anthropic-Ratelimit-Requests-Remaining": "999",
        "Anthropic-Ratelimit-Tokens-Limit": "100000",
        "Anthropic-Ratelimit-Tokens-Remaining": "99000",
    }
    if anthropic_beta:
        headers["anthropic-beta"] = anthropic_beta
    if is_stream:
        headers.update(
            {
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache",
            }
        )
    else:
        headers["Content-Type"] = "application/json"
    if conversation_id:
        headers["X-Ghost-Conversation-Id"] = conversation_id
    if parent_message_id:
        headers["X-Ghost-Parent-Message-Id"] = parent_message_id
    return headers


def _json_response(
    request: httpx.Request,
    data: Any,
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    merged_headers = {"Content-Type": "application/json", **(headers or {})}
    return httpx.Response(
        status_code=status_code,
        headers=merged_headers,
        content=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        request=request,
    )


def _openai_error(
    request: httpx.Request,
    message: str,
    *,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    code: str | None = None,
    request_id: str | None = None,
) -> httpx.Response:
    headers = _openai_headers(request_id) if request_id else {}
    return _json_response(
        request,
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
        status_code=status_code,
        headers=headers,
    )


def _anthropic_error(
    request: httpx.Request,
    message: str,
    *,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    request_id: str | None = None,
) -> httpx.Response:
    headers = _anthropic_headers(request_id) if request_id else {}
    return _json_response(
        request,
        {
            "type": "error",
            "error": {
                "type": error_type,
                "message": message,
            },
        },
        status_code=status_code,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# httpx ByteStream adapters
# ---------------------------------------------------------------------------

class _AsyncSSEByteStream(httpx.AsyncByteStream):
    def __init__(self, sse_gen: AsyncGenerator[str, None]):
        self._sse_gen = sse_gen

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._sse_gen:
            yield chunk.encode("utf-8")

    async def aclose(self) -> None:
        await self._sse_gen.aclose()


class _SyncSSEByteStream(httpx.SyncByteStream):
    def __init__(self, sync_stream: _SyncStream[str]):
        self._sync_stream = sync_stream

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._sync_stream:
                yield chunk.encode("utf-8")
        finally:
            self._sync_stream.close()

    def close(self) -> None:
        self._sync_stream.close()


# ---------------------------------------------------------------------------
# Async in-process httpx transport
# ---------------------------------------------------------------------------

class AsyncGhostTransport(httpx.AsyncBaseTransport):
    """In-process ``httpx.AsyncBaseTransport`` backed directly by ``GhostEngine``."""

    def __init__(
        self,
        *,
        engine: GhostEngine | None = None,
        registry: ProviderRegistry | None = None,
        default_provider: str | None = None,
        enable_fallback: bool = True,
        raw_logging: bool = False,
        owns_engine: bool | None = None,
    ):
        self._owns_engine = (engine is None) if owns_engine is None else owns_engine
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

    async def start(self) -> None:
        await self.engine.start()

    async def aclose(self) -> None:
        if self._owns_engine:
            await self.engine.aclose()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        method = request.method.upper()
        path = request.url.path.rstrip("/") or "/"
        raw_body = await request.aread()
        return await self._dispatch(request, method, path, raw_body, sync_runner=None)

    async def _dispatch(
        self,
        request: httpx.Request,
        method: str,
        path: str,
        raw_body: bytes,
        *,
        sync_runner: _AsyncLoopRunner | None,
    ) -> httpx.Response:
        # Health & utility endpoints
        if path in ("/health", "/api/hello") and method in ("GET", "HEAD"):
            return _json_response(request, {"status": "ok"})

        if path == "/api/event_logging/batch" and method == "POST":
            return _json_response(request, {"status": "ok"})

        # Models
        if path in ("/v1/models", "/models") and method == "GET":
            models = await self.engine.list_models()
            return _json_response(request, {"object": "list", "data": models})

        if method == "GET" and (path.startswith("/v1/models/") or path.startswith("/models/")):
            prefix = "/v1/models/" if path.startswith("/v1/models/") else "/models/"
            model_id = path[len(prefix):]
            model = await self.engine.get_model(model_id)
            if model is not None:
                return _json_response(request, model)
            return _openai_error(
                request,
                f"Model '{model_id}' not found",
                status_code=404,
                code="model_not_found",
            )

        # OpenAI Chat Completions
        if path in ("/v1/chat/completions", "/chat/completions") and method == "POST":
            return await self._handle_openai_chat(request, raw_body, sync_runner=sync_runner)

        # Anthropic token count
        if path in ("/v1/messages/count_tokens", "/messages/count_tokens") and method == "POST":
            try:
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception:
                return _json_response(request, {"input_tokens": 100})
            return _json_response(request, self.engine.count_tokens(body))

        # Anthropic Messages
        if path in ("/v1/messages", "/v1/v1/messages", "/messages") and method == "POST":
            return await self._handle_anthropic_messages(
                request, raw_body, sync_runner=sync_runner
            )

        return _openai_error(
            request,
            f"Unknown Ghost endpoint: {method} {path}",
            status_code=404,
        )

    async def _handle_openai_chat(
        self,
        request: httpx.Request,
        raw_body: bytes,
        *,
        sync_runner: _AsyncLoopRunner | None,
    ) -> httpx.Response:
        request_id = str(time.time())
        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            return _openai_error(request, "Invalid JSON", status_code=400, request_id=request_id)

        if not isinstance(body, dict):
            return _openai_error(
                request, "Request body must be a JSON object", status_code=400, request_id=request_id
            )

        if body.get("stream"):
            try:
                plan = self.engine.prepare_chat_stream(body, emit_error_chunks=True)
            except (ValidationError, ModelNotFoundError, ValueError) as exc:
                return _openai_error(request, str(exc), status_code=400, request_id=request_id)

            headers = _openai_headers(
                plan.request_id,
                conversation_id=plan.conversation_id,
                parent_message_id=plan.parent_message_id,
                is_stream=True,
            )
            stream_obj: httpx.AsyncByteStream | httpx.SyncByteStream
            if sync_runner is not None:
                stream_obj = _SyncSSEByteStream(
                    sync_runner.iter_async_gen(lambda: plan.stream)
                )
            else:
                stream_obj = _AsyncSSEByteStream(plan.stream)
            return httpx.Response(
                status_code=200, headers=headers, stream=stream_obj, request=request
            )

        try:
            result = await self.engine.chat_completion_with_metadata(body)
            headers = _openai_headers(
                result.request_id,
                elapsed_ms=result.elapsed_ms,
                conversation_id=result.conversation_id,
                parent_message_id=result.parent_message_id,
                is_stream=False,
            )
            return _json_response(request, result.data, status_code=200, headers=headers)
        except (ValidationError, ModelNotFoundError, ValueError) as exc:
            return _openai_error(request, str(exc), status_code=400, request_id=request_id)
        except RateLimitError as exc:
            return _openai_error(
                request,
                str(exc),
                status_code=429,
                error_type="rate_limit_error",
                code="rate_limit_exceeded",
                request_id=request_id,
            )
        except Exception as exc:
            return _openai_error(
                request,
                str(exc),
                status_code=500,
                error_type="server_error",
                request_id=request_id,
            )

    async def _handle_anthropic_messages(
        self,
        request: httpx.Request,
        raw_body: bytes,
        *,
        sync_runner: _AsyncLoopRunner | None,
    ) -> httpx.Response:
        request_id = str(time.time())
        try:
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            return _anthropic_error(
                request, "Invalid JSON", status_code=400, request_id=request_id
            )

        if not isinstance(body, dict):
            return _anthropic_error(
                request, "Request body must be a JSON object", status_code=400, request_id=request_id
            )

        beta_header = request.headers.get("anthropic-beta", "")

        if body.get("stream"):
            try:
                plan = self.engine.prepare_anthropic_stream(
                    body, beta_header=beta_header, emit_error_chunks=True
                )
            except (ModelNotFoundError, ValueError) as exc:
                return _anthropic_error(request, str(exc), status_code=400, request_id=request_id)

            headers = _anthropic_headers(
                plan.request_id,
                conversation_id=plan.conversation_id,
                parent_message_id=plan.parent_message_id,
                is_stream=True,
                anthropic_beta=plan.effective_beta,
            )
            stream_obj: httpx.AsyncByteStream | httpx.SyncByteStream
            if sync_runner is not None:
                stream_obj = _SyncSSEByteStream(
                    sync_runner.iter_async_gen(lambda: plan.stream)
                )
            else:
                stream_obj = _AsyncSSEByteStream(plan.stream)
            return httpx.Response(
                status_code=200, headers=headers, stream=stream_obj, request=request
            )

        try:
            result = await self.engine.anthropic_message_with_metadata(
                body, beta_header=beta_header
            )
            headers = _anthropic_headers(
                result.request_id,
                conversation_id=result.conversation_id,
                parent_message_id=result.parent_message_id,
                is_stream=False,
                anthropic_beta=result.effective_beta,
            )
            return _json_response(request, result.data, status_code=200, headers=headers)
        except (ModelNotFoundError, ValueError) as exc:
            return _anthropic_error(request, str(exc), status_code=400, request_id=request_id)
        except RateLimitError as exc:
            return _anthropic_error(
                request,
                str(exc),
                status_code=429,
                error_type="rate_limit_error",
                request_id=request_id,
            )
        except Exception as exc:
            return _anthropic_error(
                request,
                str(exc),
                status_code=500,
                error_type="api_error",
                request_id=request_id,
            )


# ---------------------------------------------------------------------------
# Sync in-process httpx transport
# ---------------------------------------------------------------------------

class GhostTransport(httpx.BaseTransport):
    """In-process ``httpx.BaseTransport`` backed directly by ``GhostEngine``."""

    def __init__(
        self,
        *,
        engine: GhostEngine | None = None,
        registry: ProviderRegistry | None = None,
        default_provider: str | None = None,
        enable_fallback: bool = True,
        warm_up: bool = False,
        raw_logging: bool = False,
        owns_engine: bool | None = None,
    ):
        self._runner = _AsyncLoopRunner()
        self._async_transport = AsyncGhostTransport(
            engine=engine,
            registry=registry,
            default_provider=default_provider,
            enable_fallback=enable_fallback,
            raw_logging=raw_logging,
            owns_engine=owns_engine,
        )
        if warm_up:
            self.start()

    @property
    def engine(self) -> GhostEngine:
        return self._async_transport.engine

    def start(self) -> None:
        self._runner.run(self._async_transport.start())

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        method = request.method.upper()
        path = request.url.path.rstrip("/") or "/"
        raw_body = request.read()
        return self._runner.run(
            self._async_transport._dispatch(
                request, method, path, raw_body, sync_runner=self._runner
            )
        )

    def close(self) -> None:
        self._runner.close(cleanup_coro_factory=self._async_transport.aclose)


# ---------------------------------------------------------------------------
# Drop-in client factories (httpx, OpenAI SDK, Anthropic SDK)
# ---------------------------------------------------------------------------

def httpx_client(
    *,
    base_url: str = DEFAULT_OPENAI_BASE_URL,
    engine: GhostEngine | None = None,
    registry: ProviderRegistry | None = None,
    default_provider: str | None = None,
    enable_fallback: bool = True,
    warm_up: bool = False,
    raw_logging: bool = False,
    **httpx_kwargs: Any,
) -> httpx.Client:
    """Create a synchronous ``httpx.Client`` talking directly to Ghost in-process."""
    transport = GhostTransport(
        engine=engine,
        registry=registry,
        default_provider=default_provider,
        enable_fallback=enable_fallback,
        warm_up=warm_up,
        raw_logging=raw_logging,
    )
    return httpx.Client(base_url=base_url, transport=transport, **httpx_kwargs)


def async_httpx_client(
    *,
    base_url: str = DEFAULT_OPENAI_BASE_URL,
    engine: GhostEngine | None = None,
    registry: ProviderRegistry | None = None,
    default_provider: str | None = None,
    enable_fallback: bool = True,
    raw_logging: bool = False,
    **httpx_kwargs: Any,
) -> httpx.AsyncClient:
    """Create an async ``httpx.AsyncClient`` talking directly to Ghost in-process."""
    transport = AsyncGhostTransport(
        engine=engine,
        registry=registry,
        default_provider=default_provider,
        enable_fallback=enable_fallback,
        raw_logging=raw_logging,
    )
    return httpx.AsyncClient(base_url=base_url, transport=transport, **httpx_kwargs)


def openai_client(
    *,
    api_key: str = "ghost",
    base_url: str = DEFAULT_OPENAI_BASE_URL,
    engine: GhostEngine | None = None,
    registry: ProviderRegistry | None = None,
    default_provider: str | None = None,
    enable_fallback: bool = True,
    warm_up: bool = False,
    raw_logging: bool = False,
    **openai_kwargs: Any,
):
    """Return a real ``openai.OpenAI`` client routed in-process through Ghost (no server)."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The official 'openai' package is required for ghost.openai_client(). "
            "Install with: pip install 'ghost[openai]' (or pip install openai)"
        ) from exc

    http_client = httpx_client(
        base_url=base_url,
        engine=engine,
        registry=registry,
        default_provider=default_provider,
        enable_fallback=enable_fallback,
        warm_up=warm_up,
        raw_logging=raw_logging,
    )
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        **openai_kwargs,
    )


def async_openai_client(
    *,
    api_key: str = "ghost",
    base_url: str = DEFAULT_OPENAI_BASE_URL,
    engine: GhostEngine | None = None,
    registry: ProviderRegistry | None = None,
    default_provider: str | None = None,
    enable_fallback: bool = True,
    raw_logging: bool = False,
    **openai_kwargs: Any,
):
    """Return a real ``openai.AsyncOpenAI`` client routed in-process through Ghost (no server)."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "The official 'openai' package is required for ghost.async_openai_client(). "
            "Install with: pip install 'ghost[openai]' (or pip install openai)"
        ) from exc

    http_client = async_httpx_client(
        base_url=base_url,
        engine=engine,
        registry=registry,
        default_provider=default_provider,
        enable_fallback=enable_fallback,
        raw_logging=raw_logging,
    )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        **openai_kwargs,
    )


def anthropic_client(
    *,
    api_key: str = "ghost",
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
    engine: GhostEngine | None = None,
    registry: ProviderRegistry | None = None,
    default_provider: str | None = None,
    enable_fallback: bool = True,
    warm_up: bool = False,
    raw_logging: bool = False,
    **anthropic_kwargs: Any,
):
    """Return a real ``anthropic.Anthropic`` client routed in-process through Ghost (no server)."""
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ImportError(
            "The official 'anthropic' package is required for ghost.anthropic_client(). "
            "Install with: pip install 'ghost[anthropic]' (or pip install anthropic)"
        ) from exc

    http_client = httpx_client(
        base_url=base_url,
        engine=engine,
        registry=registry,
        default_provider=default_provider,
        enable_fallback=enable_fallback,
        warm_up=warm_up,
        raw_logging=raw_logging,
    )
    return Anthropic(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        **anthropic_kwargs,
    )


def async_anthropic_client(
    *,
    api_key: str = "ghost",
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
    engine: GhostEngine | None = None,
    registry: ProviderRegistry | None = None,
    default_provider: str | None = None,
    enable_fallback: bool = True,
    raw_logging: bool = False,
    **anthropic_kwargs: Any,
):
    """Return a real ``anthropic.AsyncAnthropic`` client routed in-process through Ghost (no server)."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise ImportError(
            "The official 'anthropic' package is required for ghost.async_anthropic_client(). "
            "Install with: pip install 'ghost[anthropic]' (or pip install anthropic)"
        ) from exc

    http_client = async_httpx_client(
        base_url=base_url,
        engine=engine,
        registry=registry,
        default_provider=default_provider,
        enable_fallback=enable_fallback,
        raw_logging=raw_logging,
    )
    return AsyncAnthropic(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client,
        **anthropic_kwargs,
    )


__all__ = [
    "GhostTransport",
    "AsyncGhostTransport",
    "httpx_client",
    "async_httpx_client",
    "openai_client",
    "async_openai_client",
    "anthropic_client",
    "async_anthropic_client",
]
