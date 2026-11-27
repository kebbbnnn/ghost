class ProviderError(Exception):
    """Base exception for provider errors."""
    pass


class RateLimitError(ProviderError):
    """Raised when a provider rate limits the request."""
    pass


class TokenExpiredError(ProviderError):
    """Raised when a provider session/token expires and cannot be refreshed."""
    pass


class ModelNotFoundError(ProviderError, ValueError):
    """Raised when no registered provider supports the requested model."""
    pass
