"""Factory module for instantiating LLMProvider adapters."""

from __future__ import annotations

from mightycode_shared.models import ProviderConfig

from mightycode_server.providers.anthropic import AnthropicProvider
from mightycode_server.providers.base import LLMProvider
from mightycode_server.providers.errors import ProviderError
from mightycode_server.providers.gemini import GeminiProvider
from mightycode_server.providers.groq import GroqProvider
from mightycode_server.providers.ollama import OllamaProvider
from mightycode_server.providers.openai import OpenAIProvider

_PROVIDER_MAP: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "ollama": OllamaProvider,
}


def get_provider(config: ProviderConfig) -> LLMProvider:
    """Instantiate and return the appropriate LLMProvider for a given ProviderConfig.

    Args:
        config: ProviderConfig instance specifying provider name and credentials.

    Returns:
        An instance of an LLMProvider subclass.

    Raises:
        ProviderError: If the provider identifier is unknown.
    """
    provider_name = config.provider.lower().strip()
    provider_cls = _PROVIDER_MAP.get(provider_name)
    if not provider_cls:
        valid_names = ", ".join(sorted(_PROVIDER_MAP.keys()))
        msg = f"Unknown provider '{config.provider}'. Supported providers: {valid_names}"
        raise ProviderError(message=msg, provider=config.provider)

    return provider_cls(config)
