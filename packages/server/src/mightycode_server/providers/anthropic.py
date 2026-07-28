"""Anthropic LLM provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from mightycode_shared.models import (
    ChatMessage,
    MessageRole,
    ProviderConfig,
    StreamEvent,
    StreamEventType,
    ToolCall,
)

from mightycode_server.providers.base import LLMProvider
from mightycode_server.providers.errors import (
    InvalidAPIKeyError,
    ProviderAPIError,
)


def _safe_str(val: Any) -> str:
    """Extract string safely from val, returning empty string for unset mocks/Nones."""
    if isinstance(val, str):
        return val
    if val is None:
        return ""
    val_str = str(val)
    if val_str.startswith("<MagicMock") or val_str.startswith("<AsyncMock"):
        return ""
    return val_str


class AnthropicProvider(LLMProvider):
    """Adapter for Anthropic Messages API."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._validate_api_key(require_key=True)
        self.client = anthropic.AsyncAnthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url or None,
        )

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Separate system message and format conversation history for Anthropic."""
        system_prompt: str | None = None
        formatted: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content or ""
            elif msg.role == MessageRole.USER:
                formatted.append({"role": "user", "content": msg.content or ""})
            elif msg.role == MessageRole.ASSISTANT:
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                formatted.append(
                    {
                        "role": "assistant",
                        "content": content_blocks if content_blocks else msg.content or "",
                    }
                )
            elif msg.role == MessageRole.TOOL and msg.tool_result:
                formatted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_result.tool_call_id,
                                "content": msg.tool_result.content,
                                "is_error": msg.tool_result.is_error,
                            }
                        ],
                    }
                )

        return system_prompt, formatted

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert standard OpenAI-style function definitions to Anthropic tool schema."""
        if not tools:
            return None
        anthropic_tools: list[dict[str, Any]] = []
        for t in tools:
            func_def = t.get("function", t)
            schema = func_def.get("parameters", {"type": "object", "properties": {}})
            anthropic_tools.append(
                {
                    "name": func_def.get("name"),
                    "description": func_def.get("description", ""),
                    "input_schema": schema,
                }
            )
        return anthropic_tools

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        system_prompt, formatted_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": formatted_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        try:
            async with self.client.messages.stream(**kwargs) as stream:
                current_tool_id: str | None = None
                current_tool_name: str | None = None
                current_tool_input_json = ""

                async for event in stream:
                    if event.type == "text":
                        yield StreamEvent(
                            type=StreamEventType.TEXT_DELTA,
                            delta=event.text,
                        )
                    elif event.type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "tool_use":
                            current_tool_id = _safe_str(getattr(block, "id", ""))
                            current_tool_name = _safe_str(getattr(block, "name", ""))
                            current_tool_input_json = ""
                    elif event.type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            if getattr(delta, "type", None) == "text_delta":
                                yield StreamEvent(
                                    type=StreamEventType.TEXT_DELTA,
                                    delta=_safe_str(getattr(delta, "text", "")),
                                )
                            elif getattr(delta, "type", None) == "input_json_delta":
                                pj = getattr(delta, "partial_json", "")
                                current_tool_input_json += _safe_str(pj)
                    elif (
                        event.type == "content_block_stop" and current_tool_id and current_tool_name
                    ):
                        try:
                            args = (
                                json.loads(current_tool_input_json)
                                if current_tool_input_json
                                else {}
                            )
                        except json.JSONDecodeError:
                            args = {}
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL,
                            tool_call=ToolCall(
                                id=current_tool_id,
                                name=current_tool_name,
                                arguments=args,
                            ),
                        )
                        current_tool_id = None
                        current_tool_name = None
                        current_tool_input_json = ""

                # Yield final done event
                final_msg = await stream.get_final_message()
                in_tok = getattr(final_msg.usage, "input_tokens", None)
                if not isinstance(in_tok, int):
                    in_tok = None
                out_tok = getattr(final_msg.usage, "output_tokens", None)
                if not isinstance(out_tok, int):
                    out_tok = None

                stop_reason_raw = getattr(final_msg, "stop_reason", "stop")
                stop_reason = _safe_str(stop_reason_raw) or "stop"

                yield StreamEvent(
                    type=StreamEventType.DONE,
                    finish_reason=stop_reason,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )

        except (InvalidAPIKeyError, ProviderAPIError):
            raise
        except anthropic.AuthenticationError as exc:
            raise InvalidAPIKeyError(provider="anthropic", details=str(exc)) from exc
        except anthropic.APIError as exc:
            status = getattr(exc, "status_code", None)
            raise ProviderAPIError(
                provider="anthropic",
                status_code=status,
                details=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderAPIError(provider="anthropic", details=str(exc)) from exc
