import typing
from abc import ABC, abstractmethod

class BaseProvider(ABC):
    """
    Abstract base class for all AI platform providers.
    Every provider must implement these methods and return responses
    strictly conforming to the OpenAI JSON schema.
    """

    @abstractmethod
    async def get_models(self) -> list[dict]:
        """
        Return a list of supported model dicts in OpenAI format.
        Example: [{"id": "gpt-4o", "object": "model", ...}]
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> typing.AsyncGenerator[str, None]:
        """
        Stream a chat completion.
        Must yield formatted OpenAI SSE strings (e.g. `data: {"id": "...", ...}\n\n`).
        The final chunk must be `data: [DONE]\n\n`.
        """
        pass

    @abstractmethod
    async def non_streaming_chat(
        self,
        messages: list[dict],
        model: str,
        conversation_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> dict:
        """
        Return a complete OpenAI chat completion response dictionary.
        """
        pass

    async def start_background_tasks(self):
        """
        Start any background tasks, such as proactive token refresh loops or
        connection warm-ups. Base implementation does nothing.
        """
        pass

    async def stop_background_tasks(self):
        """
        Stop background tasks and clean up resources cleanly.
        Base implementation does nothing.
        """
        pass
