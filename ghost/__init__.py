"""
Ghost — Embeddable multi-provider LLM client & OpenAI/Anthropic-compatible proxy.

Use Ghost directly inside any Python project without running a server::

    from ghost import Ghost

    with Ghost() as client:
        print(client.ask("What is 2+2?", model="auto"))

Or drop directly into the official OpenAI / Anthropic SDKs in-process::

    import ghost

    oa = ghost.openai_client()      # real openai.OpenAI, zero server
    an = ghost.anthropic_client()   # real anthropic.Anthropic, zero server
"""
from ghost.adapters import (
    AsyncGhostTransport,
    GhostTransport,
    anthropic_client,
    async_anthropic_client,
    async_httpx_client,
    async_openai_client,
    httpx_client,
    openai_client,
)
from ghost.client import AsyncGhost, Ghost
from ghost.core.exceptions import (
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    TokenExpiredError,
)
from ghost.core.provider import BaseProvider
from ghost.engine import ChatCompletionRequest, ChatMessage, GhostEngine
from ghost.providers import ProviderRegistry

__version__ = "2.1.0"

__all__ = [
    "__version__",
    "Ghost",
    "AsyncGhost",
    "GhostEngine",
    "ProviderRegistry",
    "BaseProvider",
    "ChatMessage",
    "ChatCompletionRequest",
    "GhostTransport",
    "AsyncGhostTransport",
    "httpx_client",
    "async_httpx_client",
    "openai_client",
    "async_openai_client",
    "anthropic_client",
    "async_anthropic_client",
    "ProviderError",
    "RateLimitError",
    "TokenExpiredError",
    "ModelNotFoundError",
]
