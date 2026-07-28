"""Groq LLM provider adapter (OpenAI-compatible endpoint)."""

from __future__ import annotations

from mightycode_shared.models import ProviderConfig

from mightycode_server.providers.openai import OpenAIProvider


class GroqProvider(OpenAIProvider):
    """Adapter for Groq API (OpenAI-compatible endpoint)."""

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, config: ProviderConfig) -> None:
        # Guarantee base_url defaults to Groq's endpoint if not supplied
        effective_config = config.model_copy(
            update={
                "provider": "groq",
                "base_url": config.base_url or self.DEFAULT_BASE_URL,
            }
        )
        super().__init__(effective_config)
