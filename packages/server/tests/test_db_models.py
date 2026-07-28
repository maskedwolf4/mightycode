"""End-to-end database tests against a real Neon database.

These tests require ``DATABASE_URL`` and ``MIGHTYCODE_ENCRYPTION_KEY``
environment variables to be set. They create real rows, read them back,
and verify FK relationships and encryption.
"""

from __future__ import annotations

import os
import uuid

import pytest
import sqlalchemy.exc
from cryptography.fernet import Fernet
from mightycode_server.encryption import decrypt_value, encrypt_value, reset_fernet_cache
from mightycode_server.models import (
    AgentRun,
    Message,
    ProviderKey,
    Session,
    ToolCallLog,
    UsageEvent,
    User,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

_TEST_ENCRYPTION_KEY = Fernet.generate_key().decode()


@pytest.fixture(autouse=True)
def _set_encryption_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the encryption key env var is set for all tests."""
    monkeypatch.setenv("MIGHTYCODE_ENCRYPTION_KEY", _TEST_ENCRYPTION_KEY)
    reset_fernet_cache()


def _get_test_db_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        pytest.skip("DATABASE_URL not set – skipping DB tests")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


@pytest.fixture
async def db_session() -> AsyncSession:  # type: ignore[misc]
    """Create a fresh async session that rolls back after each test."""
    url = _get_test_db_url()
    engine = create_async_engine(url, echo=False)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session  # type: ignore[misc]
    await engine.dispose()


class TestUserSessionMessage:
    """Core e2e test: create User → Session → Message and read back."""

    @pytest.mark.asyncio
    async def test_create_and_read_back(self, db_session: AsyncSession) -> None:
        unique = uuid.uuid4().hex[:8]

        user = User(email=f"test-{unique}@example.com", display_name="Test User")
        db_session.add(user)
        await db_session.flush()

        session = Session(user_id=user.id, title="Test Session")
        db_session.add(session)
        await db_session.flush()

        msg = Message(
            session_id=session.id,
            role="user",
            content="Hello, MightyCode!",
        )
        db_session.add(msg)
        await db_session.flush()

        loaded_msg = await db_session.get(Message, msg.id)
        assert loaded_msg is not None
        assert loaded_msg.content == "Hello, MightyCode!"
        assert loaded_msg.role == "user"
        assert loaded_msg.session_id == session.id

        loaded_session = await db_session.get(Session, session.id)
        assert loaded_session is not None
        assert loaded_session.user_id == user.id

        loaded_user = await db_session.get(User, user.id)
        assert loaded_user is not None
        assert loaded_user.email == f"test-{unique}@example.com"

        await db_session.rollback()


class TestForeignKeyRelationships:
    """Verify FK constraints are enforced."""

    @pytest.mark.asyncio
    async def test_message_requires_valid_session(self, db_session: AsyncSession) -> None:
        fake_session_id = uuid.uuid4()
        msg = Message(session_id=fake_session_id, role="user", content="orphan")
        db_session.add(msg)
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            await db_session.flush()
        await db_session.rollback()

    @pytest.mark.asyncio
    async def test_usage_event_fk_to_user(self, db_session: AsyncSession) -> None:
        unique = uuid.uuid4().hex[:8]
        user = User(email=f"usage-{unique}@example.com")
        db_session.add(user)
        await db_session.flush()

        event = UsageEvent(
            user_id=user.id,
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
        )
        db_session.add(event)
        await db_session.flush()

        loaded = await db_session.get(UsageEvent, event.id)
        assert loaded is not None
        assert loaded.user_id == user.id
        assert loaded.provider == "openai"

        await db_session.rollback()

    @pytest.mark.asyncio
    async def test_agent_run_self_referential_fk(self, db_session: AsyncSession) -> None:
        unique = uuid.uuid4().hex[:8]
        user = User(email=f"agent-{unique}@example.com")
        db_session.add(user)
        await db_session.flush()

        session = Session(user_id=user.id, title="Agent Test")
        db_session.add(session)
        await db_session.flush()

        parent = AgentRun(session_id=session.id, status="completed")
        db_session.add(parent)
        await db_session.flush()

        child = AgentRun(
            session_id=session.id,
            parent_run_id=parent.id,
            status="running",
        )
        db_session.add(child)
        await db_session.flush()

        loaded_child = await db_session.get(AgentRun, child.id)
        assert loaded_child is not None
        assert loaded_child.parent_run_id == parent.id

        await db_session.rollback()

    @pytest.mark.asyncio
    async def test_tool_call_log_fk_to_message(self, db_session: AsyncSession) -> None:
        unique = uuid.uuid4().hex[:8]
        user = User(email=f"tool-{unique}@example.com")
        db_session.add(user)
        await db_session.flush()

        session = Session(user_id=user.id)
        db_session.add(session)
        await db_session.flush()

        msg = Message(session_id=session.id, role="assistant", content=None)
        db_session.add(msg)
        await db_session.flush()

        log = ToolCallLog(
            message_id=msg.id,
            tool_name="read_file",
            arguments={"path": "/tmp/test"},
            result="file contents",
            is_error=False,
            duration_ms=42,
        )
        db_session.add(log)
        await db_session.flush()

        loaded = await db_session.get(ToolCallLog, log.id)
        assert loaded is not None
        assert loaded.tool_name == "read_file"
        assert loaded.message_id == msg.id

        await db_session.rollback()


class TestProviderKeyEncryption:
    """Verify ProviderKey stores API keys encrypted, never plaintext."""

    @pytest.mark.asyncio
    async def test_api_key_encrypted_at_rest(self, db_session: AsyncSession) -> None:
        unique = uuid.uuid4().hex[:8]
        user = User(email=f"enc-{unique}@example.com")
        db_session.add(user)
        await db_session.flush()

        pk = ProviderKey(user_id=user.id, provider="openai", encrypted_api_key="")
        pk.set_api_key("sk-secret-test-key-12345")
        db_session.add(pk)
        await db_session.flush()

        loaded = await db_session.get(ProviderKey, pk.id)
        assert loaded is not None
        assert loaded.encrypted_api_key != "sk-secret-test-key-12345"
        assert loaded.encrypted_api_key.startswith("gAAAAA")
        assert loaded.get_api_key() == "sk-secret-test-key-12345"

        await db_session.rollback()

    def test_encrypt_decrypt_roundtrip(self) -> None:
        original = "sk-my-secret-api-key"
        encrypted = encrypt_value(original)
        assert encrypted != original
        assert decrypt_value(encrypted) == original

    def test_encrypted_value_is_not_plaintext(self) -> None:
        secret = "super-secret-password"
        encrypted = encrypt_value(secret)
        assert secret not in encrypted


class TestModelDefaults:
    """Verify default values and auto-generated fields."""

    @pytest.mark.asyncio
    async def test_user_timestamps(self, db_session: AsyncSession) -> None:
        unique = uuid.uuid4().hex[:8]
        user = User(email=f"ts-{unique}@example.com")
        db_session.add(user)
        await db_session.flush()

        loaded = await db_session.get(User, user.id)
        assert loaded is not None
        assert loaded.created_at is not None
        assert loaded.updated_at is not None

        await db_session.rollback()

    @pytest.mark.asyncio
    async def test_agent_run_default_status(self, db_session: AsyncSession) -> None:
        unique = uuid.uuid4().hex[:8]
        user = User(email=f"stat-{unique}@example.com")
        db_session.add(user)
        await db_session.flush()

        session = Session(user_id=user.id)
        db_session.add(session)
        await db_session.flush()

        run = AgentRun(session_id=session.id)
        db_session.add(run)
        await db_session.flush()

        loaded = await db_session.get(AgentRun, run.id)
        assert loaded is not None
        assert loaded.status == "pending"

        await db_session.rollback()
