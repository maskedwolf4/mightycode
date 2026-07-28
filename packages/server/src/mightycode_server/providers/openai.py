"""OpenAI LLM provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import openai
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


class OpenAIProvider(LLMProvider):
    """Adapter for OpenAI Chat Completions API."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._validate_api_key(require_key=True)
        self.client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or None,
        )

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Format ChatMessage objects to OpenAI API standard message format."""
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                formatted.append({"role": "system", "content": msg.content or ""})
            elif msg.role == MessageRole.USER:
                formatted.append({"role": "user", "content": msg.content or ""})
            elif msg.role == MessageRole.ASSISTANT:
                item: dict[str, Any] = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                formatted.append(item)
            elif msg.role == MessageRole.TOOL and msg.tool_result:
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_result.tool_call_id,
                        "content": msg.tool_result.content,
                    }
                )
        return formatted

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        formatted_messages = self._convert_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        try:
            stream = await self.client.chat.completions.create(**kwargs)
            pending_tool_calls: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None
            input_tokens: int | None = None
            output_tokens: int | None = None

            async for chunk in stream:
                if hasattr(chunk, "usage") and chunk.usage:
                    pt = getattr(chunk.usage, "prompt_tokens", None)
                    if isinstance(pt, int):
                        input_tokens = pt
                    ct = getattr(chunk.usage, "completion_tokens", None)
                    if isinstance(ct, int):
                        output_tokens = ct

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                fr = getattr(choice, "finish_reason", None)
                if isinstance(fr, str):
                    finish_reason = fr

                delta = getattr(choice, "delta", None)
                if delta and getattr(delta, "content", None):
                    yield StreamEvent(
                        type=StreamEventType.TEXT_DELTA,
                        delta=_safe_str(delta.content),
                    )

                if delta and getattr(delta, "tool_calls", None):
                    for tc_delta in delta.tool_calls:
                        idx = getattr(tc_delta, "index", 0)
                        fn = getattr(tc_delta, "function", None)
                        tc_id = _safe_str(getattr(tc_delta, "id", ""))
                        fn_name = _safe_str(getattr(fn, "name", "")) if fn else ""
                        fn_args = _safe_str(getattr(fn, "arguments", "")) if fn else ""

                        if idx not in pending_tool_calls:
                            pending_tool_calls[idx] = {
                                "id": tc_id,
                                "name": fn_name,
                                "arguments_json": fn_args,
                            }
                        else:
                            if tc_id:
                                pending_tool_calls[idx]["id"] = tc_id
                            if fn_name:
                                pending_tool_calls[idx]["name"] += fn_name
                            if fn_args:
                                pending_tool_calls[idx]["arguments_json"] += fn_args

            # Yield tool calls accumulated during streaming
            for tc_data in pending_tool_calls.values():
                try:
                    raw_args = tc_data["arguments_json"]
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
                yield StreamEvent(
                    type=StreamEventType.TOOL_CALL,
                    tool_call=ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=args,
                    ),
                )

            # Yield completion done event
            yield StreamEvent(
                type=StreamEventType.DONE,
                finish_reason=finish_reason or "stop",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except (InvalidAPIKeyError, ProviderAPIError):
            raise
        except openai.AuthenticationError as exc:
            raise InvalidAPIKeyError(provider=self.config.provider, details=str(exc)) from exc
        except openai.APIError as exc:
            raise ProviderAPIError(
                provider=self.config.provider,
                status_code=getattr(exc, "status_code", None),
                details=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderAPIError(provider=self.config.provider, details=str(exc)) from exc
