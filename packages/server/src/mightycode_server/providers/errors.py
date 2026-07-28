"""Typed exception hierarchy for LLM providers.

Ensures provider-specific errors (missing keys, authentication failures, API errors)
are caught and wrapped into clean, typed exceptions with zero raw exception leakage.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base exception for all provider errors."""

    def __init__(self, message: str, provider: str = "unknown") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider


class MissingAPIKeyError(ProviderError):
    """Raised when a required API key is missing or empty."""

    def __init__(self, provider: str) -> None:
        message = f"API key for provider '{provider}' is missing or empty."
        super().__init__(message, provider=provider)


class InvalidAPIKeyError(ProviderError):
    """Raised when an API key is rejected (401 / authentication failure)."""

    def __init__(self, provider: str, details: str = "") -> None:
        message = f"API key for provider '{provider}' is invalid."
        if details:
            message += f" Details: {details}"
        super().__init__(message, provider=provider)


class ProviderAPIError(ProviderError):
    """Raised when a provider API encounters an HTTP or operational failure."""

    def __init__(self, provider: str, status_code: int | None = None, details: str = "") -> None:
        message = f"Provider '{provider}' API request failed"
        if status_code is not None:
            message += f" with status {status_code}"
        if details:
            message += f": {details}"
        super().__init__(message, provider=provider)
        self.status_code = status_code
