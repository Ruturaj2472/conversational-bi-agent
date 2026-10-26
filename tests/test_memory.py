import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import memory


@pytest.fixture(autouse=True)
def sqlite_memory(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    memory.Base.metadata.create_all(engine)
    monkeypatch.setattr(memory, "_engine", engine)
    monkeypatch.setattr(memory, "_SessionLocal", sessionmaker(bind=engine))
    yield
    engine.dispose()


def test_store_messages_keeps_order_and_skips_empty():
    inserted = memory.store_messages(
        "s1", [("user", "Top genre?"), ("assistant", "Rock."), ("assistant", "  ")]
    )
    assert inserted == 2
    history = memory.get_history("s1")
    assert [(h["role"], h["content"]) for h in history] == [
        ("user", "Top genre?"),
        ("assistant", "Rock."),
    ]


def test_store_messages_rejects_bad_role():
    with pytest.raises(ValueError):
        memory.store_messages("s1", [("system", "hi")])


def test_history_returns_latest_n_oldest_first():
    for i in range(6):
        memory.store_message("s1", "user", f"q{i}")
    memory.store_message("other", "user", "not mine")
    history = memory.get_history("s1", limit=3)
    assert [h["content"] for h in history] == ["q3", "q4", "q5"]


def test_langchain_history_truncates_long_answers():
    memory.store_messages("s1", [("user", "q"), ("assistant", "x" * 50)])
    msgs = memory.get_history_as_langchain_messages("s1", max_chars=10)
    assert msgs[0].type == "human" and msgs[0].content == "q"
    assert msgs[1].type == "ai" and msgs[1].content == "x" * 10 + "…"


def test_clear_history():
    memory.store_messages("s1", [("user", "q"), ("assistant", "a")])
    assert memory.clear_history("s1") == 2
    assert memory.get_history("s1") == []
