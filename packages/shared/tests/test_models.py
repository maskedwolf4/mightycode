"""Tests for the shared Pydantic models.

These tests prove the contracts are importable, serialise correctly,
and enforce validation rules.
"""

from __future__ import annotations

import json

import pytest
from mightycode_shared import (
    ChatMessage,
    MessageRole,
    ProviderConfig,
    ToolCall,
    ToolResult,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_create_minimal(self) -> None:
        tc = ToolCall(id="tc_1", name="read_file")
        assert tc.id == "tc_1"
        assert tc.name == "read_file"
        assert tc.arguments == {}

    def test_create_with_arguments(self) -> None:
        tc = ToolCall(id="tc_2", name="write_file", arguments={"path": "/tmp/x", "data": "hi"})
        assert tc.arguments["path"] == "/tmp/x"

    def test_roundtrip_json(self) -> None:
        tc = ToolCall(id="tc_3", name="search", arguments={"query": "hello"})
        raw = tc.model_dump_json()
        restored = ToolCall.model_validate_json(raw)
        assert restored == tc


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success_result(self) -> None:
        tr = ToolResult(tool_call_id="tc_1", content="file contents here")
        assert tr.is_error is False

    def test_error_result(self) -> None:
        tr = ToolResult(tool_call_id="tc_1", content="not found", is_error=True)
        assert tr.is_error is True

    def test_roundtrip_json(self) -> None:
        tr = ToolResult(tool_call_id="tc_2", content="ok", is_error=False)
        restored = ToolResult.model_validate_json(tr.model_dump_json())
        assert restored == tr


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_user_message(self) -> None:
        msg = ChatMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.tool_calls == []
        assert msg.tool_result is None

    def test_assistant_with_tool_calls(self) -> None:
        tc = ToolCall(id="tc_1", name="read_file", arguments={"path": "main.py"})
        msg = ChatMessage(role=MessageRole.ASSISTANT, content=None, tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "read_file"

    def test_tool_result_message(self) -> None:
        tr = ToolResult(tool_call_id="tc_1", content="file data")
        msg = ChatMessage(role=MessageRole.TOOL, tool_result=tr)
        assert msg.role == MessageRole.TOOL
        assert msg.tool_result is not None
        assert msg.tool_result.content == "file data"

    def test_roundtrip_json(self) -> None:
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Sure, let me read that.",
            tool_calls=[ToolCall(id="tc_5", name="grep", arguments={"pattern": "TODO"})],
        )
        data = json.loads(msg.model_dump_json())
        restored = ChatMessage.model_validate(data)
        assert restored == msg

    def test_all_roles_valid(self) -> None:
        for role in MessageRole:
            msg = ChatMessage(role=role, content="test")
            assert msg.role == role


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_minimal_config(self) -> None:
        cfg = ProviderConfig(provider="openai", model="gpt-4o")
        assert cfg.api_key == ""
        assert cfg.base_url is None
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 4096

    def test_full_config(self) -> None:
        cfg = ProviderConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="sk-test",
            base_url="https://proxy.example.com",
            temperature=0.7,
            max_tokens=8192,
        )
        assert cfg.provider == "anthropic"
        assert cfg.base_url == "https://proxy.example.com"

    def test_temperature_validation(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(provider="x", model="y", temperature=3.0)

    def test_max_tokens_validation(self) -> None:
        with pytest.raises(ValidationError):
            ProviderConfig(provider="x", model="y", max_tokens=0)

    def test_roundtrip_json(self) -> None:
        cfg = ProviderConfig(provider="ollama", model="llama3", base_url="http://localhost:11434")
        restored = ProviderConfig.model_validate_json(cfg.model_dump_json())
        assert restored == cfg
