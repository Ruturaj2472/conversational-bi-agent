"""
End-to-end run of the LangGraph agent with a scripted fake LLM and SQLite,
so the graph wiring can be tested without OpenAI or PostgreSQL.
"""

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from app import agent as agent_module


class ScriptedChatModel(BaseChatModel):
    """Returns pre-scripted AIMessages / structured outputs in order."""

    replies: list = []
    structured: dict = {}
    calls: list = []
    stored: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        def respond(_messages: Any):
            self.calls.append(schema.__name__)
            return schema(**self.structured[schema.__name__].pop(0))

        return RunnableLambda(respond)


@pytest.fixture
def sqlite_engine():
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE genre (name TEXT, tracks INTEGER)"))
        conn.execute(text("INSERT INTO genre VALUES ('Rock', 1297), ('Jazz', 130)"))
    return eng


@pytest.fixture
def model(monkeypatch, sqlite_engine):
    m = ScriptedChatModel(replies=[], structured={}, calls=[], stored=[])
    monkeypatch.setattr(agent_module, "init_chat_model", lambda *a, **k: m)
    monkeypatch.setattr(agent_module, "create_engine", lambda *a, **k: sqlite_engine)
    monkeypatch.setattr(
        agent_module, "get_history_as_langchain_messages", lambda *a, **k: []
    )
    monkeypatch.setattr(
        agent_module, "store_messages", lambda sid, msgs: m.stored.extend(msgs)
    )
    return m


def tool_call(sql: str, call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "sql_db_query", "args": {"query": sql}, "id": call_id}],
    )


def test_chart_question_runs_sql_and_builds_chart_config(model):
    model.structured = {
        "QuestionAnalysis": [{"rewritten_question": "Tracks per genre?", "wants_chart": True}],
        "ChartConfig": [
            {
                "chart_type": "bar",
                "x_col": "name",
                "y_col": "tracks",
                "title": "Tracks per genre",
                "x_label": "Genre",
                "y_label": "Tracks",
            }
        ],
    }
    model.replies = [
        tool_call("SELECT name, tracks FROM genre ORDER BY tracks DESC"),
        AIMessage(content="Rock has the most tracks.\nCHART_REQUEST"),
    ]
    graph, _db = agent_module.build_agent()
    result = agent_module.run_agent(graph, "How many tracks per genre? chart", "s1")

    assert result["answer"] == "Rock has the most tracks."
    assert result["sql"].startswith("SELECT name, tracks")
    assert result["sql_columns"] == ["name", "tracks"]
    assert result["sql_rows"] == [("Rock", 1297), ("Jazz", 130)]
    assert result["wants_chart"] is True
    assert result["chart_config"]["chart_type"] == "bar"
    assert result["rewritten_q"] == "Tracks per genre?"
    assert model.stored == [
        ("user", "How many tracks per genre? chart"),
        ("assistant", "Rock has the most tracks."),
    ]
    # analyze + generate(tool) + generate(answer) + chart = 4 model calls
    assert len(model.calls) == 4


def test_schema_is_in_system_prompt(model):
    model.structured = {
        "QuestionAnalysis": [{"rewritten_question": "Count genres", "wants_chart": False}]
    }
    model.replies = [AIMessage(content="There are 2 genres.")]
    graph, _db = agent_module.build_agent()
    result = agent_module.run_agent(graph, "Count genres", "s1")

    assert result["answer"] == "There are 2 genres."
    assert result["chart_config"] is None
    system_prompt = model.calls[1][0].content
    assert "CREATE TABLE genre" in system_prompt


def test_sql_error_is_retried(model):
    model.structured = {
        "QuestionAnalysis": [{"rewritten_question": "Top genre", "wants_chart": False}]
    }
    model.replies = [
        tool_call("SELECT nme FROM genre", "call_1"),
        tool_call("SELECT name FROM genre ORDER BY tracks DESC LIMIT 1", "call_2"),
        AIMessage(content="Rock."),
    ]
    graph, _db = agent_module.build_agent()
    result = agent_module.run_agent(graph, "Top genre", "s1")

    assert result["answer"] == "Rock."
    assert result["sql_rows"] == [("Rock",)]
    error_tool_msg = model.calls[2][-1]
    assert error_tool_msg.type == "tool" and error_tool_msg.content.startswith("Error:")


def test_gives_up_after_recursion_limit(model):
    model.structured = {
        "QuestionAnalysis": [{"rewritten_question": "Bad", "wants_chart": False}]
    }
    model.replies = [tool_call("SELECT broken", f"c{i}") for i in range(20)]
    graph, _db = agent_module.build_agent()
    result = agent_module.run_agent(graph, "Bad", "s1")

    assert result["answer"] == agent_module.GIVE_UP_ANSWER


def test_chart_skipped_when_no_rows(model):
    model.structured = {
        "QuestionAnalysis": [{"rewritten_question": "Pop tracks", "wants_chart": True}]
    }
    model.replies = [
        tool_call("SELECT name, tracks FROM genre WHERE name = 'Pop'"),
        AIMessage(content="No Pop tracks found."),
    ]
    graph, _db = agent_module.build_agent()
    result = agent_module.run_agent(graph, "Pop tracks chart", "s1")

    assert result["chart_config"] is None
    assert "ChartConfig" not in model.calls
