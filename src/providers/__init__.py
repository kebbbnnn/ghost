from .chatgpt.session import ChatGPTProvider
from .gemini.session import GeminiProvider
from .spectre.session import SpectreProvider

# The ProviderRegistry holds instances of all available providers.
_chatgpt = ChatGPTProvider()
_gemini = GeminiProvider()
_spectre = SpectreProvider()

PROVIDERS = [
    _chatgpt,
    _gemini,
    _spectre,
]

# Map provider classes to their instances for routing
_PROVIDER_MAP = {type(p): p for p in PROVIDERS}


async def get_all_models() -> list[dict]:
    """Aggregate models from all registered providers."""
    all_models = []
    for p in PROVIDERS:
        try:
            models = await p.get_models()
            all_models.extend(models)
        except Exception:
            pass
    return all_models


def get_provider_for_model(model_name: str):
    """
    Find the provider that supports the given model.
    Routes based on model name prefix:
      - spectre*    → SpectreProvider
      - gemini*     → GeminiProvider
      - gpt-*, auto, or anything else → ChatGPTProvider (default)
    """
    name = model_name.lower().strip()

    if name.startswith("spectre"):
        return _PROVIDER_MAP.get(SpectreProvider)

    if name.startswith("gemini"):
        return _PROVIDER_MAP.get(GeminiProvider)

    # Default to ChatGPT for gpt-*, auto, and unknown models
    return _PROVIDER_MAP.get(ChatGPTProvider)


