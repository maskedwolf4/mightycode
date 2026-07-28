"""Pydantic models shared across MightyCode CLI and Server.

These contracts are the canonical source of truth for data crossing
package boundaries. Do NOT use ad-hoc dicts – always serialise via
these models.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MessageRole(StrEnum):
    """Role of a participant in a chat conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class StreamEventType(StrEnum):
    """Normalized stream event types across all LLM providers."""

    TEXT_DELTA = "text_delta"
    TOOL_CALL = "tool_call"
    DONE = "done"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Core chat models
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A request from the model to invoke a tool."""

    id: str = Field(..., description="Unique identifier for this tool invocation.")
    name: str = Field(..., description="Name of the tool to call.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serialisable arguments passed to the tool.",
    )


class ToolResult(BaseModel):
    """The result returned after executing a tool call."""

    tool_call_id: str = Field(..., description="ID of the ToolCall this result answers.")
    content: str = Field(..., description="Textual output of the tool execution.")
    is_error: bool = Field(
        default=False,
        description="Whether the tool execution ended in an error.",
    )


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: MessageRole = Field(..., description="Who sent this message.")
    content: str | None = Field(
        default=None,
        description="Text body of the message (may be None for tool-call-only messages).",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Tool invocations requested by the assistant.",
    )
    tool_result: ToolResult | None = Field(
        default=None,
        description="Result payload when role == TOOL.",
    )


# ---------------------------------------------------------------------------
# Stream Event model
# ---------------------------------------------------------------------------


class StreamEvent(BaseModel):
    """Normalized stream event yielded during LLM provider streaming responses."""

    type: StreamEventType = Field(..., description="Type of stream event.")
    delta: str | None = Field(default=None, description="Text chunk for text_delta events.")
    tool_call: ToolCall | None = Field(
        default=None, description="Tool call object for tool_call events."
    )
    finish_reason: str | None = Field(
        default=None, description="Completion reason for done events."
    )
    input_tokens: int | None = Field(default=None, description="Number of input tokens processed.")
    output_tokens: int | None = Field(
        default=None, description="Number of output tokens generated."
    )
    error_type: str | None = Field(
        default=None, description="Error classification string for error events."
    )
    message: str | None = Field(
        default=None, description="Human-readable message for error events."
    )


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider (loaded from env / config file).

    Secrets are **never** hardcoded; they come from environment variables
    or ``~/.mightycode/config.toml``.
    """

    provider: str = Field(
        ...,
        description="Provider identifier, e.g. 'openai', 'anthropic', 'ollama'.",
    )
    model: str = Field(
        ...,
        description="Model name to use, e.g. 'gpt-4o', 'claude-sonnet-4-20250514'.",
    )
    api_key: str = Field(
        default="",
        description=(
            "API key for the provider. "
            "Should be populated from env var or config file, never hardcoded."
        ),
    )
    base_url: str | None = Field(
        default=None,
        description="Optional custom base URL for self-hosted / proxy endpoints.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens in the response.",
    )
