"""
app/sql_runner.py - Read-only SQL execution for the agent

Runs the agent's SQL through SQLAlchemy and returns the real column names
and JSON-friendly row values.

Public API:
    execute_readonly(engine, sql, max_rows) -> QueryResult
"""

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    error: Optional[str] = None

    def to_tool_content(self) -> str:
        """Compact text the LLM reads as the tool result."""
        if self.error:
            return f"Error: {self.error}"
        payload = {"columns": self.columns, "rows": [list(r) for r in self.rows]}
        if self.truncated:
            payload["note"] = f"Only the first {len(self.rows)} rows are shown."
        return json.dumps(payload, default=str)


def _to_plain(value: Any) -> Any:
    """Convert DB-specific types to JSON / pandas friendly values."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (bytes, memoryview)):
        return "<binary>"
    return value


def execute_readonly(engine: Engine, sql: str, max_rows: int = 100) -> QueryResult:
    """
    Execute one SQL statement inside a read-only transaction.

    Write statements fail at the database level on PostgreSQL. The transaction
    is always rolled back, and at most `max_rows` rows are fetched.
    """
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return QueryResult(error="Empty query.")

    options = {}
    if engine.dialect.name == "postgresql":
        options["postgresql_readonly"] = True

    try:
        with engine.connect().execution_options(**options) as conn:
            try:
                result = conn.exec_driver_sql(sql)
                if not result.returns_rows:
                    return QueryResult()
                columns = list(result.keys())
                fetched = result.fetchmany(max_rows + 1)
            finally:
                conn.rollback()
    except Exception as exc:
        message = str(getattr(exc, "orig", None) or exc).strip()
        logger.info(f"execute_readonly error: {message[:200]}")
        return QueryResult(error=message)

    truncated = len(fetched) > max_rows
    rows = [tuple(_to_plain(v) for v in row) for row in fetched[:max_rows]]
    return QueryResult(columns=columns, rows=rows, truncated=truncated)
