"""
Provider registry & model routing.

Provides ``ProviderRegistry`` for isolated per-client provider lifecycles, plus
a module-level default registry (``PROVIDERS``, ``get_all_models``,
``get_provider_for_model``, ``get_fallback_provider``) for server and
backwards-compatibility usage.
"""
from __future__ import annotations

import os
from typing import Optional

from ghost.core.provider import BaseProvider
from .chatgpt.session import ChatGPTProvider
from .gemini.session import GeminiProvider
from .spectre.session import SpectreProvider


class ProviderRegistry:
    """Isolated collection of provider instances owned by a single Ghost client/engine."""

    def __init__(
        self,
        *,
        default_provider: str | None = None,
        chatgpt: ChatGPTProvider | None = None,
        gemini: GeminiProvider | None = None,
        spectre: SpectreProvider | None = None,
    ):
        self.default_provider = (default_provider or "").lower().strip() or None
        self.chatgpt = chatgpt if chatgpt is not None else ChatGPTProvider()
        self.gemini = gemini if gemini is not None else GeminiProvider()
        self.spectre = (
            spectre
            if spectre is not None
            else SpectreProvider(provider_resolver=self.get_provider_for_model)
        )
        self.providers: list[BaseProvider] = [
            self.chatgpt,
            self.gemini,
            self.spectre,
        ]
        self._provider_map: dict[type[BaseProvider], BaseProvider] = {
            type(p): p for p in self.providers
        }

    async def get_all_models(self) -> list[dict]:
        """Aggregate models from all registered providers."""
        all_models: list[dict] = []
        for provider in self.providers:
            try:
                models = await provider.get_models()
                all_models.extend(models)
            except Exception:
                pass
        return all_models

    def get_provider_for_model(self, model_name: str) -> Optional[BaseProvider]:
        """
        Find the provider that supports the given model.
        Routes based on model name prefix:
          - spectre*            -> SpectreProvider
          - gemini*             -> GeminiProvider
          - claude*, anthropic* -> GeminiProvider (handles large context window required by Claude Code)
          - gpt-*, auto, or anything else -> ChatGPTProvider (or configured default_provider)
        """
        name = model_name.lower().strip()

        if name.startswith("spectre"):
            return self._provider_map.get(SpectreProvider, self.spectre)

        if name.startswith("gemini") or name.startswith("claude") or name.startswith("anthropic"):
            return self._provider_map.get(GeminiProvider, self.gemini)

        default_prov = (
            self.default_provider
            if self.default_provider is not None
            else os.environ.get("GHOST_DEFAULT_PROVIDER", "").lower().strip()
        )
        if default_prov == "gemini":
            return self._provider_map.get(GeminiProvider, self.gemini)

        return self._provider_map.get(ChatGPTProvider, self.chatgpt)

    def get_fallback_provider(self, current_provider: BaseProvider) -> Optional[BaseProvider]:
        """Return an alternate provider when current_provider encounters an error."""
        if current_provider is self.gemini:
            return self.chatgpt
        if current_provider is self.chatgpt:
            return self.gemini
        if current_provider is self.spectre:
            return self.gemini
        return None

    async def start_background_tasks(self) -> None:
        """Start background refresh/warm-up tasks for all providers."""
        for provider in self.providers:
            await provider.start_background_tasks()

    async def stop_background_tasks(self) -> None:
        """Stop background tasks for all providers."""
        for provider in self.providers:
            await provider.stop_background_tasks()

    async def aclose(self) -> None:
        """Stop background tasks and close all underlying provider HTTP sessions."""
        await self.stop_background_tasks()
        for provider in self.providers:
            try:
                await provider.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Default module-level registry (used by ghost.server & legacy callers)
# ---------------------------------------------------------------------------

_default_registry = ProviderRegistry()
_chatgpt = _default_registry.chatgpt
_gemini = _default_registry.gemini
_spectre = _default_registry.spectre

PROVIDERS = _default_registry.providers
_PROVIDER_MAP = _default_registry._provider_map


async def get_all_models() -> list[dict]:
    """Aggregate models from all registered providers in the default registry."""
    return await _default_registry.get_all_models()


def get_provider_for_model(model_name: str):
    """Route a model name using the default registry."""
    return _default_registry.get_provider_for_model(model_name)


def get_fallback_provider(current_provider):
    """Return a fallback provider using the default registry."""
    return _default_registry.get_fallback_provider(current_provider)


__all__ = [
    "ProviderRegistry",
    "ChatGPTProvider",
    "GeminiProvider",
    "SpectreProvider",
    "PROVIDERS",
    "get_all_models",
    "get_provider_for_model",
    "get_fallback_provider",
]
