"""
mlflow/tracker.py - MLflow experiment tracking for the BI Agent

Fully non-blocking — if MLflow server is unreachable (e.g. running locally
without Docker), all tracking calls are silently skipped and the app
continues normally. No retries, no startup delays.

Each query is logged through MlflowClient with an explicit run_id, so
concurrent requests never share or close each other's runs.

To use MLflow locally:
    pip install mlflow
    mlflow ui --port 5000
Then set MLFLOW_TRACKING_URI=http://localhost:5000 in your .env
"""

import logging
import os
import time
import threading
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Module-level state ────────────────────────────────────────────────────────
_mlflow_enabled: bool = False  # set to True only if init succeeds
_client = None  # MlflowClient, created once the server is reachable
_experiment_id: Optional[str] = None


def init_mlflow() -> None:
    """
    Initialise MLflow tracking in a background thread so it never blocks startup.

    If the server is unreachable the thread will fail silently after one
    connection attempt (no long retry loop in the foreground).
    LangChain autologging is enabled only once the server is confirmed reachable.
    """
    try:
        import mlflow  # noqa: F401
    except ImportError:
        logger.warning("mlflow package not installed — tracking disabled.")
        return

    def _init():
        global _mlflow_enabled, _client, _experiment_id
        # Fail fast instead of MLflow's default multi-minute retry/backoff
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            experiment = mlflow.set_experiment(settings.mlflow_experiment_name)
            mlflow.langchain.autolog()

            _client = MlflowClient()
            _experiment_id = experiment.experiment_id
            _mlflow_enabled = True
            logger.info(
                f"MLflow connected: {settings.mlflow_tracking_uri} | "
                f"experiment={settings.mlflow_experiment_name}"
            )
        except Exception as exc:
            _mlflow_enabled = False
            logger.warning(
                f"MLflow unavailable (tracking disabled): {exc}. "
                f"Start MLflow with: mlflow ui --port 5000"
            )

    # Run in daemon thread — dies with the process, never blocks main startup
    t = threading.Thread(target=_init, daemon=True, name="mlflow-init")
    t.start()


# ── QueryTracker ──────────────────────────────────────────────────────────────


class QueryTracker:
    """
    Context manager that tracks one agent query in MLflow.
    Every method is fully guarded — if MLflow is down or disabled,
    the body of the `with` block runs normally with zero overhead.

    Usage:
        with track_query(session_id="user-123", question="...") as tracker:
            result = await run_agent(...)
            tracker.log_result(result)
    """

    def __init__(self, session_id: str, question: str):
        self.session_id = session_id
        self.question = question
        self._run_id: Optional[str] = None
        self._start: float = 0.0
        self._chart_generated = False

    def __enter__(self) -> "QueryTracker":
        self._start = time.perf_counter()
        if not _mlflow_enabled or _client is None:
            return self
        try:
            run = _client.create_run(
                experiment_id=_experiment_id,
                run_name=f"query_{self.session_id[:8]}",
                tags={
                    "session_id": self.session_id,
                    "model": settings.model_name,
                    "component": "sql-agent",
                },
            )
            self._run_id = run.info.run_id
            _client.log_param(self._run_id, "question", self.question[:500])
            _client.log_param(self._run_id, "session_id", self.session_id)
            _client.log_param(self._run_id, "model", settings.model_name)
        except Exception as exc:
            logger.debug(f"MLflow create_run skipped: {exc}")
            self._run_id = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = round(time.perf_counter() - self._start, 3)
        if not self._run_id:
            return
        try:
            _client.log_metric(self._run_id, "latency_seconds", elapsed)
            _client.log_metric(
                self._run_id, "chart_generated", 1 if self._chart_generated else 0
            )
            if exc_type:
                _client.set_tag(self._run_id, "status", "error")
                _client.log_param(self._run_id, "error", str(exc_val)[:500])
                _client.set_terminated(self._run_id, status="FAILED")
            else:
                _client.set_tag(self._run_id, "status", "success")
                _client.set_terminated(self._run_id, status="FINISHED")
        except Exception as exc:
            logger.debug(f"MLflow set_terminated skipped: {exc}")

    def log_result(self, result: dict) -> None:
        """Log answer metrics — no-op if MLflow is disabled."""
        if not self._run_id:
            return
        try:
            answer = result.get("answer", "")
            messages = result.get("raw_messages", [])

            _client.log_metric(self._run_id, "answer_length_chars", len(answer))
            _client.log_metric(self._run_id, "message_count", len(messages))
            _client.log_metric(self._run_id, "sql_rows", len(result.get("sql_rows", [])))
            _client.log_metric(
                self._run_id, "wants_chart", 1 if result.get("wants_chart") else 0
            )
            if answer:
                _client.log_param(self._run_id, "answer_preview", answer[:300])
            if result.get("sql"):
                _client.log_param(self._run_id, "sql", result["sql"][:500])
        except Exception as exc:
            logger.debug(f"MLflow log_result skipped: {exc}")

    def log_chart(self, chart_result: dict) -> None:
        """Log chart metrics — no-op if MLflow is disabled."""
        self._chart_generated = True
        if not self._run_id:
            return
        try:
            _client.log_metric(self._run_id, "chart_rows", chart_result.get("rows", 0))
            _client.log_param(
                self._run_id, "chart_type", chart_result.get("chart_type", "unknown")
            )
            _client.set_tag(
                self._run_id, "chart_title", chart_result.get("title", "")[:200]
            )
        except Exception as exc:
            logger.debug(f"MLflow log_chart skipped: {exc}")


def track_query(session_id: str, question: str) -> QueryTracker:
    """Return a QueryTracker context manager for one agent turn."""
    return QueryTracker(session_id=session_id, question=question)
