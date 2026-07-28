"""Abstract base class for LLM Provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from mightycode_shared.models import ChatMessage, ProviderConfig, StreamEvent

from mightycode_server.providers.errors import MissingAPIKeyError


class LLMProvider(ABC):
    """Abstract base class for all LLM provider adapters.

    All implementations MUST normalize their raw API streaming responses
    into standard ``StreamEvent`` instances without leaking provider-specific details.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def _validate_api_key(self, require_key: bool = True) -> None:
        """Verify API key is present when required."""
        if require_key and not self.config.api_key.strip():
            raise MissingAPIKeyError(provider=self.config.provider)

    @abstractmethod
    def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a chat completion response yielded as standard StreamEvent items.

        Args:
            messages: Conversation history as list of ChatMessage instances.
            tools: Optional tool definitions in standard JSON-schema format.

        Yields:
            StreamEvent objects (text_delta, tool_call, done, error).
        """
        ...
