"""
app/memory.py - Custom SQLAlchemy-based conversation memory

Replaces the LangGraph checkpointer with a transparent, hand-rolled
conversation store backed by PostgreSQL via SQLAlchemy.

Schema (auto-created on startup):
    Table: conversation_messages
        id          SERIAL PRIMARY KEY
        session_id  VARCHAR(255) NOT NULL
        role        VARCHAR(50)  NOT NULL   -- 'user' | 'assistant'
        content     TEXT         NOT NULL
        created_at  TIMESTAMP    DEFAULT NOW()

Public API:
    init_db()                                      -- create engine + tables
    store_message(session_id, role, content)        -- INSERT one message
    get_history(session_id, limit)                  -- SELECT last N rows as dicts
    get_history_as_langchain_messages(session_id)   -- returns LangChain message objects
    get_recent_context_string(session_id, limit)    -- formatted string for prompts
    clear_history(session_id)                       -- DELETE all rows for session
    close_db()                                      -- dispose engine
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── ORM ───────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class ConversationMessage(Base):
    """One turn in a conversation — a single user or assistant message."""

    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<ConversationMessage id={self.id} session={self.session_id!r} "
            f"role={self.role!r} len={len(self.content)}>"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ── Engine / session factory ──────────────────────────────────────────────────

_engine = None
_SessionLocal: Optional[sessionmaker] = None


def init_db() -> None:
    """
    Create the SQLAlchemy engine, run CREATE TABLE IF NOT EXISTS,
    and set up the session factory.

    Call once at application startup (in FastAPI lifespan).
    """
    global _engine, _SessionLocal

    db_url = settings.memory_db_url  # postgresql+psycopg2://...
    safe_url = db_url.split("@")[-1]  # strip credentials for logging
    logger.info(f"Initialising memory DB at: {safe_url}")

    _engine = create_engine(
        db_url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # detect stale connections automatically
        echo=settings.debug,  # log SQL in debug mode
    )

    # Create tables — safe to call repeatedly (IF NOT EXISTS)
    Base.metadata.create_all(_engine)

    _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    logger.info("Memory DB ready — table 'conversation_messages' ensured.")


def close_db() -> None:
    """Dispose the engine connection pool. Call at application shutdown."""
    global _engine
    if _engine:
        _engine.dispose()
        logger.info("Memory DB connection pool disposed.")


def _get_session() -> Session:
    """Internal: return a new SQLAlchemy session."""
    if _SessionLocal is None:
        raise RuntimeError("Memory DB not initialised. Call init_db() first.")
    return _SessionLocal()


# ── Write ─────────────────────────────────────────────────────────────────────


def store_message(session_id: str, role: str, content: str) -> ConversationMessage:
    """
    Persist one message to PostgreSQL.

    Args:
        session_id: Conversation thread identifier (e.g. "user-abc-123").
        role:       'user' or 'assistant'.
        content:    Plain-text message body.

    Returns:
        The saved ConversationMessage ORM object (with id and created_at set).

    Raises:
        ValueError: if role is not 'user'/'assistant' or content is empty.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
    if not content or not content.strip():
        raise ValueError("content must not be empty")

    row = ConversationMessage(
        session_id=session_id,
        role=role,
        content=content.strip(),
    )
    with _get_session() as db:
        db.add(row)
        db.commit()
        db.refresh(row)
        logger.debug(f"Stored → {row}")
        return row


# ── Read ──────────────────────────────────────────────────────────────────────


def get_history(
    session_id: str,
    limit: int = 20,
) -> list[dict]:
    """
    Fetch the last `limit` messages for a session, returned oldest-first.

    Args:
        session_id: Conversation thread identifier.
        limit:      Maximum number of rows (default 20 ≈ 10 turns).

    Returns:
        List of dicts: [{"id", "session_id", "role", "content", "created_at"}, ...]
    """
    with _get_session() as db:
        rows = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.session_id == session_id)
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        # Reverse so the list is chronological (oldest → newest)
        return [r.to_dict() for r in reversed(rows)]


def get_history_as_langchain_messages(
    session_id: str,
    limit: int = 20,
    max_chars: Optional[int] = None,
) -> list[BaseMessage]:
    """
    Fetch conversation history as LangChain HumanMessage / AIMessage objects,
    ready to be prepended to the agent's message list.

    Args:
        session_id: Conversation thread identifier.
        limit:      Maximum number of messages (default 20).
        max_chars:  If set, assistant messages longer than this are truncated.

    Returns:
        Chronological list of BaseMessage objects.
    """
    history = get_history(session_id, limit=limit)
    messages: list[BaseMessage] = []
    for row in history:
        content = row["content"]
        if row["role"] == "user":
            messages.append(HumanMessage(content=content))
        else:
            if max_chars and len(content) > max_chars:
                content = content[:max_chars] + "…"
            messages.append(AIMessage(content=content))
    return messages


def get_recent_context_string(
    session_id: str,
    limit: int = 10,
) -> str:
    """
    Return recent history as a compact plain-text block for injection into
    system prompts (e.g. the query rewriter).

    Format:
        User: which genre has the longest tracks?
        Assistant: Sci Fi & Fantasy, averaging 2911 seconds per track.
        User: what about the shortest?

    Long assistant answers are truncated to 400 chars to keep prompts tight.

    Args:
        session_id: Conversation thread identifier.
        limit:      Number of recent messages (default 10).

    Returns:
        Formatted string, or empty string when there is no history yet.
    """
    history = get_history(session_id, limit=limit)
    if not history:
        return ""

    lines = []
    for row in history:
        label = "User" if row["role"] == "user" else "Assistant"
        content = row["content"]
        if row["role"] == "assistant" and len(content) > 400:
            content = content[:400] + "…"
        lines.append(f"{label}: {content}")

    return "\n".join(lines)


# ── Delete ────────────────────────────────────────────────────────────────────


def clear_history(session_id: str) -> int:
    """
    Delete all messages for a given session.

    Args:
        session_id: Conversation thread identifier.

    Returns:
        Number of rows deleted.
    """
    with _get_session() as db:
        deleted = (
            db.query(ConversationMessage)
            .filter(ConversationMessage.session_id == session_id)
            .delete(synchronize_session=False)
        )
        db.commit()
        logger.info(f"Cleared {deleted} messages for session '{session_id}'.")
        return deleted
