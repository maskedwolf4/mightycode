"""Smoke tests for the CLI entry point."""

from __future__ import annotations

import subprocess
import sys


class TestCLISmoke:
    def test_smoke_flag_exits_cleanly(self) -> None:
        """``mightycode --smoke`` must exit 0 and print the probe message."""
        result = subprocess.run(
            [sys.executable, "-m", "mightycode_cli.main", "--smoke"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "smoke OK" in result.stdout

    def test_shared_import_from_cli(self) -> None:
        """Shared models must be importable from within the cli package."""
        from mightycode_shared import (
            ChatMessage,
            MessageRole,
            ProviderConfig,
            ToolCall,
            ToolResult,
        )

        msg = ChatMessage(role=MessageRole.ASSISTANT, content="hi")
        assert msg.role == MessageRole.ASSISTANT
        tc = ToolCall(id="tc_1", name="test")
        assert tc.name == "test"
        tr = ToolResult(tool_call_id="tc_1", content="ok")
        assert not tr.is_error
        cfg = ProviderConfig(provider="test", model="m")
        assert cfg.provider == "test"
