"""
Integration tests against a real PostgreSQL with the Northwind data loaded.

Skipped unless INTEGRATION_DB_URL (northwind) is set; memory tests also need
INTEGRATION_MEMORY_DB_URL (agent_memory). No LLM calls.
"""

import os
import pathlib
import uuid
from types import SimpleNamespace

import pytest
import yaml
from sqlalchemy import create_engine, text

from app.sql_runner import execute_readonly

DB_URL = os.getenv("INTEGRATION_DB_URL")
MEMORY_DB_URL = os.getenv("INTEGRATION_MEMORY_DB_URL")
QUESTIONS = pathlib.Path(__file__).parents[2] / "evals" / "questions.yaml"

pytestmark = pytest.mark.skipif(not DB_URL, reason="INTEGRATION_DB_URL not set")


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(DB_URL)
    yield eng
    eng.dispose()


def test_writes_are_rejected_by_the_database(engine):
    with engine.connect() as conn:
        before = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
    result = execute_readonly(engine, "DELETE FROM products")
    assert result.error and "read-only" in result.error
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM products")).scalar() == before


def test_ddl_is_rejected_by_the_database(engine):
    result = execute_readonly(engine, "DROP TABLE products")
    assert result.error and "read-only" in result.error


def test_statement_timeout_stops_slow_queries():
    eng = create_engine(DB_URL, connect_args={"options": "-c statement_timeout=500"})
    result = execute_readonly(eng, "SELECT pg_sleep(2)")
    eng.dispose()
    assert result.error and "statement timeout" in result.error


def test_numeric_and_date_values_are_converted(engine):
    result = execute_readonly(
        engine, "SELECT unit_price, order_date FROM order_details JOIN orders USING (order_id) LIMIT 1"
    )
    price, order_date = result.rows[0]
    assert isinstance(price, float)
    assert isinstance(order_date, str) and order_date[:2] == "19"


def test_row_cap_and_truncation_flag(engine):
    result = execute_readonly(engine, "SELECT * FROM order_details", max_rows=100)
    assert len(result.rows) == 100 and result.truncated


@pytest.mark.parametrize(
    "spec",
    [q for q in yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))["questions"] if q.get("reference_sql")],
    ids=lambda q: q["id"],
)
def test_reference_sql_runs(engine, spec):
    result = execute_readonly(engine, spec["reference_sql"], max_rows=1000)
    assert result.error is None, result.error
    assert result.rows


@pytest.mark.skipif(not MEMORY_DB_URL, reason="INTEGRATION_MEMORY_DB_URL not set")
def test_memory_round_trip_and_index(monkeypatch):
    from app import memory

    monkeypatch.setattr(memory, "settings", SimpleNamespace(memory_db_url=MEMORY_DB_URL, debug=False))
    memory.init_db()
    session = f"it-{uuid.uuid4().hex[:8]}"
    try:
        assert memory.store_messages(session, [("user", "q1"), ("assistant", "a1")]) == 2
        memory.store_message(session, "user", "q2")
        assert [h["content"] for h in memory.get_history(session, limit=2)] == ["a1", "q2"]

        with memory._engine.connect() as conn:
            indexes = conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = 'conversation_messages'")
            ).scalars().all()
        assert "ix_conversation_messages_session_id_id" in indexes
    finally:
        memory.clear_history(session)
        memory.close_db()
