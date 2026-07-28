"""Tests for LLM provider adapters normalization and error handling."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mightycode_server.providers import (
    AnthropicProvider,
    GeminiProvider,
    GroqProvider,
    InvalidAPIKeyError,
    LLMProvider,
    MissingAPIKeyError,
    OllamaProvider,
    OpenAIProvider,
    get_provider,
)
from mightycode_shared.models import (
    ChatMessage,
    MessageRole,
    ProviderConfig,
    StreamEvent,
    StreamEventType,
    ToolCall,
)


class TestProviderInterfaceAndFactory:
    """Verify provider adapters subclass LLMProvider and factory instantiates correctly."""

    def test_factory_instantiates_all_providers(self) -> None:
        providers = [
            ("anthropic", "claude-3-5-sonnet", AnthropicProvider, "sk-ant-123"),
            ("openai", "gpt-4o", OpenAIProvider, "sk-openai-123"),
            ("gemini", "gemini-1.5-pro", GeminiProvider, "AIzaSy123"),
            ("groq", "llama-3.3-70b", GroqProvider, "gsk_123"),
            ("ollama", "llama3", OllamaProvider, ""),
        ]
        for p_name, model, expected_cls, api_key in providers:
            cfg = ProviderConfig(provider=p_name, model=model, api_key=api_key)
            provider = get_provider(cfg)
            assert isinstance(provider, expected_cls)
            assert isinstance(provider, LLMProvider)
            assert hasattr(provider, "stream_chat")

    def test_missing_api_key_raises_typed_error(self) -> None:
        for p_name in ["anthropic", "openai", "gemini", "groq"]:
            cfg = ProviderConfig(provider=p_name, model="test-model", api_key="")
            with pytest.raises(MissingAPIKeyError) as exc_info:
                get_provider(cfg)
            assert exc_info.value.provider == p_name
            assert "missing or empty" in str(exc_info.value)

    def test_ollama_allows_empty_api_key(self) -> None:
        cfg = ProviderConfig(provider="ollama", model="llama3", api_key="")
        provider = get_provider(cfg)
        assert isinstance(provider, OllamaProvider)


class TestProviderStreamNormalization:
    """Mock raw provider API responses and assert identical StreamEvent outputs."""

    @pytest.fixture
    def sample_messages(self) -> list[ChatMessage]:
        return [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful coding assistant."),
            ChatMessage(role=MessageRole.USER, content="Read the README file."),
        ]

    @pytest.fixture
    def sample_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read file contents",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_openai_stream_normalization(
        self, sample_messages: list[ChatMessage], sample_tools: list[dict[str, Any]]
    ) -> None:
        cfg = ProviderConfig(provider="openai", model="gpt-4o", api_key="sk-test")
        provider = OpenAIProvider(cfg)

        chunk1 = MagicMock()
        chunk1.choices = [
            MagicMock(delta=MagicMock(content="Hello", tool_calls=None), finish_reason=None)
        ]
        chunk1.usage = None

        chunk2 = MagicMock()
        tc_func = MagicMock()
        tc_func.name = "read_file"
        tc_func.arguments = '{"path": "README.md"}'

        tc_delta = MagicMock(index=0, id="call_abc123", function=tc_func)
        chunk2.choices = [
            MagicMock(delta=MagicMock(content=None, tool_calls=[tc_delta]), finish_reason=None)
        ]
        chunk2.usage = None

        chunk3 = MagicMock()
        chunk3.choices = [
            MagicMock(delta=MagicMock(content=None, tool_calls=None), finish_reason="stop")
        ]
        chunk3.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        async def mock_generator() -> Any:
            for c in [chunk1, chunk2, chunk3]:
                yield c

        with patch.object(
            provider.client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_generator(),
        ):
            events = [e async for e in provider.stream_chat(sample_messages, sample_tools)]

        assert len(events) == 3
        assert events[0] == StreamEvent(type=StreamEventType.TEXT_DELTA, delta="Hello")
        assert events[1] == StreamEvent(
            type=StreamEventType.TOOL_CALL,
            tool_call=ToolCall(id="call_abc123", name="read_file", arguments={"path": "README.md"}),
        )
        assert events[2] == StreamEvent(
            type=StreamEventType.DONE, finish_reason="stop", input_tokens=10, output_tokens=5
        )

    @pytest.mark.asyncio
    async def test_anthropic_stream_normalization(
        self, sample_messages: list[ChatMessage], sample_tools: list[dict[str, Any]]
    ) -> None:
        cfg = ProviderConfig(provider="anthropic", model="claude-3-5-sonnet", api_key="sk-ant-test")
        provider = AnthropicProvider(cfg)

        evt_text = MagicMock(type="text", text="Hello")

        cb_mock = MagicMock()
        cb_mock.type = "tool_use"
        cb_mock.id = "toolu_123"
        cb_mock.name = "read_file"

        block_start = MagicMock(type="content_block_start", content_block=cb_mock)

        delta_mock = MagicMock()
        delta_mock.type = "input_json_delta"
        delta_mock.partial_json = '{"path": "README.md"}'

        block_delta = MagicMock(type="content_block_delta", delta=delta_mock)
        block_stop = MagicMock(type="content_block_stop")

        mock_stream = AsyncMock()

        async def mock_stream_iter() -> Any:
            for item in [evt_text, block_start, block_delta, block_stop]:
                yield item

        mock_stream.__aiter__ = lambda s: mock_stream_iter()
        mock_final_msg = MagicMock(
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=15, output_tokens=8),
        )
        mock_stream.get_final_message = AsyncMock(return_value=mock_final_msg)

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__.return_value = mock_stream

        mock_messages_stream = MagicMock(return_value=mock_stream_ctx)

        with patch.object(provider.client.messages, "stream", mock_messages_stream):
            events = [e async for e in provider.stream_chat(sample_messages, sample_tools)]

        assert len(events) == 3
        assert events[0] == StreamEvent(type=StreamEventType.TEXT_DELTA, delta="Hello")
        assert events[1] == StreamEvent(
            type=StreamEventType.TOOL_CALL,
            tool_call=ToolCall(id="toolu_123", name="read_file", arguments={"path": "README.md"}),
        )
        assert events[2] == StreamEvent(
            type=StreamEventType.DONE, finish_reason="end_turn", input_tokens=15, output_tokens=8
        )

    @pytest.mark.asyncio
    async def test_gemini_stream_normalization(
        self, sample_messages: list[ChatMessage], sample_tools: list[dict[str, Any]]
    ) -> None:
        cfg = ProviderConfig(provider="gemini", model="gemini-1.5-pro", api_key="AIzaTest")
        provider = GeminiProvider(cfg)

        lines = [
            b',{"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}\n',
            (
                b',{"candidates": [{"content": {"parts": [{"functionCall": {"name": "read_file",'
                b' "args": {"path": "README.md"}}}]}}]}\n'
            ),
            (
                b',{"candidates": [{"finishReason": "STOP"}], '
                b'"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 6}}\n'
            ),
        ]

        async def mock_aiter_lines() -> Any:
            for line in lines:
                yield line.decode("utf-8")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__.return_value = mock_response

        with patch("httpx.AsyncClient.stream", return_value=mock_stream_ctx):
            events = [e async for e in provider.stream_chat(sample_messages, sample_tools)]

        assert len(events) == 3
        assert events[0] == StreamEvent(type=StreamEventType.TEXT_DELTA, delta="Hello")
        assert events[1] == StreamEvent(
            type=StreamEventType.TOOL_CALL,
            tool_call=ToolCall(id="read_file", name="read_file", arguments={"path": "README.md"}),
        )
        assert events[2] == StreamEvent(
            type=StreamEventType.DONE, finish_reason="stop", input_tokens=12, output_tokens=6
        )

    @pytest.mark.asyncio
    async def test_groq_and_ollama_stream_normalization(
        self, sample_messages: list[ChatMessage]
    ) -> None:
        groq_cfg = ProviderConfig(provider="groq", model="llama-3.3-70b", api_key="gsk-test")
        groq_provider = GroqProvider(groq_cfg)
        assert groq_provider.config.base_url == "https://api.groq.com/openai/v1"

        ollama_cfg = ProviderConfig(provider="ollama", model="llama3", api_key="")
        ollama_provider = OllamaProvider(ollama_cfg)
        assert ollama_provider.config.base_url == "http://localhost:11434/v1"


class TestProviderErrorHandling:
    """Verify raw API exceptions map to clean typed ProviderErrors."""

    @pytest.mark.asyncio
    async def test_openai_invalid_key_error(self) -> None:
        import openai

        cfg = ProviderConfig(provider="openai", model="gpt-4o", api_key="invalid-key")
        provider = OpenAIProvider(cfg)

        auth_err = openai.AuthenticationError(
            message="Incorrect API key provided",
            response=MagicMock(status_code=401),
            body=None,
        )

        user_msg = ChatMessage(role=MessageRole.USER, content="hi")
        with patch.object(provider.client.chat.completions, "create", side_effect=auth_err):
            with pytest.raises(InvalidAPIKeyError) as exc_info:
                _ = [e async for e in provider.stream_chat([user_msg])]

            assert exc_info.value.provider == "openai"
            assert "invalid" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_anthropic_invalid_key_error(self) -> None:
        import anthropic

        cfg = ProviderConfig(provider="anthropic", model="claude-3-5-sonnet", api_key="invalid-key")
        provider = AnthropicProvider(cfg)

        auth_err = anthropic.AuthenticationError(
            message="Invalid x-api-key",
            response=MagicMock(status_code=401),
            body=None,
        )

        user_msg = ChatMessage(role=MessageRole.USER, content="hi")
        with patch.object(provider.client.messages, "stream", side_effect=auth_err):
            with pytest.raises(InvalidAPIKeyError) as exc_info:
                _ = [e async for e in provider.stream_chat([user_msg])]

            assert exc_info.value.provider == "anthropic"
            assert "invalid" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_gemini_invalid_key_error(self) -> None:
        cfg = ProviderConfig(provider="gemini", model="gemini-1.5-pro", api_key="invalid-key")
        provider = GeminiProvider(cfg)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.aread = AsyncMock(return_value=b'{"error": "API key not valid"}')

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__.return_value = mock_response

        user_msg = ChatMessage(role=MessageRole.USER, content="hi")
        with patch("httpx.AsyncClient.stream", return_value=mock_stream_ctx):
            with pytest.raises(InvalidAPIKeyError) as exc_info:
                _ = [e async for e in provider.stream_chat([user_msg])]

            assert exc_info.value.provider == "gemini"
            assert "invalid" in str(exc_info.value).lower()
