"""Ollama LLM provider adapter (local models, optional API key)."""

from __future__ import annotations

import openai
from mightycode_shared.models import ProviderConfig

from mightycode_server.providers.base import LLMProvider
from mightycode_server.providers.openai import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """Adapter for local Ollama instance (OpenAI-compatible /v1 endpoint)."""

    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(self, config: ProviderConfig) -> None:
        effective_config = config.model_copy(
            update={
                "provider": "ollama",
                "base_url": config.base_url or self.DEFAULT_BASE_URL,
                "api_key": config.api_key or "ollama",
            }
        )
        LLMProvider.__init__(self, effective_config)
        self._validate_api_key(require_key=False)

        self.client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
