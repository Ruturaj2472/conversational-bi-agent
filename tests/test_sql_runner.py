import datetime as dt
import json
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, text

from app.sql_runner import _to_plain, execute_readonly


@pytest.fixture
def engine():
    eng = create_engine("sqlite://")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE genre (name TEXT, tracks INTEGER)"))
        conn.execute(
            text("INSERT INTO genre VALUES ('Rock', 1297), ('Jazz', 130), ('Blues', 81)")
        )
    return eng


def test_returns_real_column_names_and_rows(engine):
    res = execute_readonly(
        engine,
        "SELECT name AS genre, ROUND(tracks * 1.0, 2) AS track_count "
        "FROM genre ORDER BY tracks DESC",
    )
    assert res.error is None
    assert res.columns == ["genre", "track_count"]
    assert res.rows[0] == ("Rock", 1297.0)


def test_truncates_to_max_rows(engine):
    res = execute_readonly(engine, "SELECT * FROM genre", max_rows=2)
    assert len(res.rows) == 2
    assert res.truncated
    assert "Only the first 2 rows" in res.to_tool_content()


def test_sql_error_is_returned_not_raised(engine):
    res = execute_readonly(engine, "SELECT * FROM missing_table")
    assert res.error
    assert res.to_tool_content().startswith("Error:")


def test_empty_query():
    res = execute_readonly(create_engine("sqlite://"), "  ;")
    assert res.error == "Empty query."


def test_changes_are_rolled_back(engine):
    execute_readonly(engine, "DELETE FROM genre")
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM genre")).scalar() == 3


def test_tool_content_is_json(engine):
    content = execute_readonly(engine, "SELECT name FROM genre LIMIT 1").to_tool_content()
    assert json.loads(content) == {"columns": ["name"], "rows": [["Rock"]]}


@pytest.mark.parametrize(
    "value, expected",
    [
        (Decimal("1.99"), 1.99),
        (dt.date(2024, 1, 31), "2024-01-31"),
        (dt.datetime(2024, 1, 31, 9, 30), "2024-01-31T09:30:00"),
        (b"\x00", "<binary>"),
        ("text", "text"),
        (None, None),
    ],
)
def test_to_plain(value, expected):
    assert _to_plain(value) == expected
