"""Tests for the MightyCode FastAPI server."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from mightycode_server.main import app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient]:
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestServerSmoke:
    @pytest.mark.asyncio
    async def test_root(self, client: AsyncClient) -> None:
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data

    @pytest.mark.asyncio
    async def test_health(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["probe_role"] == "system"

    @pytest.mark.asyncio
    async def test_shared_import_from_server(self, client: AsyncClient) -> None:
        """Shared models must be importable from within the server package."""
        from mightycode_shared import ProviderConfig, ToolCall

        cfg = ProviderConfig(provider="test", model="test-model")
        tc = ToolCall(id="tc_1", name="noop")
        assert cfg.provider == "test"
        assert tc.name == "noop"
