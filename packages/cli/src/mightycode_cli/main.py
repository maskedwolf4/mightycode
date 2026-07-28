"""MightyCode CLI entry point.

Boots a minimal Textual application. Phase 1 is a stub that proves the
Textual app starts and the shared models are importable.
"""

from __future__ import annotations

import sys

from mightycode_shared import ChatMessage, MessageRole
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Static


class MightyCodeApp(App[None]):
    """Phase-1 stub of the MightyCode terminal UI."""

    TITLE = "MightyCode"
    SUB_TITLE = "AI Coding Agent"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            "[bold green]MightyCode CLI[/bold green] v0.1.0\n\n"
            "Phase 1 scaffold – the agent loop is not implemented yet.\n"
            "Press [bold]q[/bold] to quit.",
            id="splash",
        )
        yield Footer()

    def on_mount(self) -> None:
        # Prove shared models are importable at boot time.
        _probe = ChatMessage(role=MessageRole.USER, content="probe")
        self.log(f"Shared model probe OK: {_probe.role.value}")


def main() -> None:
    """CLI entry point invoked by ``mightycode`` console script."""
    # Allow a quick smoke-test exit for CI / scripting.
    if "--smoke" in sys.argv:
        _probe = ChatMessage(role=MessageRole.USER, content="smoke")
        print(f"mightycode-cli: smoke OK (shared probe: {_probe.role.value})")
        raise SystemExit(0)

    app = MightyCodeApp()
    app.run()


if __name__ == "__main__":
    main()
