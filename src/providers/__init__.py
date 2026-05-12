from .chatgpt.session import ChatGPTProvider

# The ProviderRegistry holds instances of all available providers.
# In the future, this can dynamically load providers from this directory.
PROVIDERS = [
    ChatGPTProvider(),
]

async def get_all_models() -> list[dict]:
    """Aggregate models from all registered providers."""
    all_models = []
    for p in PROVIDERS:
        try:
            models = await p.get_models()
            all_models.extend(models)
        except Exception as e:
            # log warning
            pass
    return all_models

def get_provider_for_model(model_name: str):
    """
    Find the provider that supports the given model.
    For now, we simply return ChatGPTProvider as it is the only one.
    Later, we can query `p.get_models()` to dynamically route.
    """
    # Hardcode routing to ChatGPTProvider for now, since it supports "auto", "gpt-*" etc.
    return PROVIDERS[0]
