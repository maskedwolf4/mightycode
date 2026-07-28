"""MightyCode Server – FastAPI application.

Phase 1: minimal health-check endpoint that proves the server boots
and can import the shared Pydantic models.
"""

from __future__ import annotations

from fastapi import FastAPI
from mightycode_shared import ChatMessage, MessageRole

app = FastAPI(
    title="MightyCode Server",
    version="0.1.0",
    description="Backend API for the MightyCode terminal coding agent.",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe – also proves shared models are importable."""
    _probe = ChatMessage(role=MessageRole.SYSTEM, content="health-check")
    return {"status": "ok", "probe_role": _probe.role.value}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"message": "MightyCode Server v0.1.0 – not implemented yet"}
