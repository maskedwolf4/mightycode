"""Google Gemini LLM provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
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


class GeminiProvider(LLMProvider):
    """Adapter for Google Gemini API."""

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._validate_api_key(require_key=True)
        self.base_url = (config.base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def _convert_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert ChatMessage objects to Gemini contents format.

        Returns the system instruction separately because Gemini expects it in
        a dedicated request field rather than inline conversation contents.
        """
        contents: list[dict[str, Any]] = []
        tool_names_by_id: dict[str, str] = {}
        system_prompt: str | None = None

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content or ""
                continue
            if msg.role == MessageRole.USER:
                contents.append({"role": "user", "parts": [{"text": msg.content or ""}]})
            elif msg.role == MessageRole.ASSISTANT:
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    tool_names_by_id[tc.id] = tc.name
                    parts.append(
                        {
                            "functionCall": {
                                "name": tc.name,
                                "args": tc.arguments,
                            }
                        }
                    )
                contents.append({"role": "model", "parts": parts})
            elif msg.role == MessageRole.TOOL and msg.tool_result:
                tool_name = tool_names_by_id.get(
                    msg.tool_result.tool_call_id, msg.tool_result.tool_call_id
                )
                contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": tool_name,
                                    "response": {"output": msg.tool_result.content},
                                }
                            }
                        ],
                    }
                )
        return system_prompt, contents

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Convert OpenAI-style tool schemas to Gemini functionDeclarations format."""
        if not tools:
            return None
        declarations: list[dict[str, Any]] = []
        for t in tools:
            func_def = t.get("function", t)
            declarations.append(
                {
                    "name": func_def.get("name"),
                    "description": func_def.get("description", ""),
                    "parameters": func_def.get("parameters", {"type": "OBJECT", "properties": {}}),
                }
            )
        return [{"functionDeclarations": declarations}]

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        system_prompt, contents = self._convert_messages(messages)
        gemini_tools = self._convert_tools(tools)

        model_name = self.config.model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        url = f"{self.base_url}/{model_name}:streamGenerateContent?key={self.config.api_key}"

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if gemini_tools:
            payload["tools"] = gemini_tools

        try:
            generated_tool_index = 0
            async with (
                httpx.AsyncClient(timeout=60.0) as client,
                client.stream("POST", url, json=payload) as response,
            ):
                if response.status_code in (401, 403):
                    body = await response.aread()
                    raise InvalidAPIKeyError(provider="gemini", details=body.decode())
                if response.status_code >= 400:
                    body = await response.aread()
                    raise ProviderAPIError(
                        provider="gemini",
                        status_code=response.status_code,
                        details=body.decode(),
                    )

                async for line in response.aiter_lines():
                    line_str = line.strip()
                    if not line_str or line_str.startswith("[") or line_str.startswith("]"):
                        continue
                    if line_str.startswith(","):
                        line_str = line_str[1:].strip()
                    if not line_str:
                        continue

                    try:
                        data = json.loads(line_str)
                    except json.JSONDecodeError:
                        continue

                    candidates = data.get("candidates", [])
                    if not candidates:
                        continue

                    candidate = candidates[0]
                    parts = candidate.get("content", {}).get("parts", [])
                    for part in parts:
                        if "text" in part:
                            yield StreamEvent(
                                type=StreamEventType.TEXT_DELTA,
                                delta=part["text"],
                            )
                        elif "functionCall" in part:
                            fc = part["functionCall"]
                            tool_id = fc.get("id")
                            if not tool_id:
                                tool_id = f"gemini_tool_{generated_tool_index}"
                                generated_tool_index += 1
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL,
                                tool_call=ToolCall(
                                    id=tool_id,
                                    name=fc.get("name", ""),
                                    arguments=fc.get("args", {}),
                                ),
                            )

                    usage = data.get("usageMetadata", {})
                    input_tokens = usage.get("promptTokenCount")
                    output_tokens = usage.get("candidatesTokenCount")
                    finish_reason = candidate.get("finishReason", "STOP").lower()

            yield StreamEvent(
                type=StreamEventType.DONE,
                finish_reason=finish_reason if "finish_reason" in locals() else "stop",
                input_tokens=input_tokens if "input_tokens" in locals() else None,
                output_tokens=output_tokens if "output_tokens" in locals() else None,
            )

        except (InvalidAPIKeyError, ProviderAPIError):
            raise
        except Exception as exc:
            raise ProviderAPIError(provider="gemini", details=str(exc)) from exc
