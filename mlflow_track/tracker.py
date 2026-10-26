"""
mlflow/tracker.py - MLflow experiment tracking for the BI Agent

Fully non-blocking — if MLflow server is unreachable (e.g. running locally
without Docker), all tracking calls are silently skipped and the app
continues normally. No retries, no startup delays.

To use MLflow locally:
    pip install mlflow
    mlflow ui --port 5000
Then set MLFLOW_TRACKING_URI=http://localhost:5000 in your .env
"""

import logging
import time
import threading
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Module-level state ────────────────────────────────────────────────────────
_mlflow_enabled: bool = False  # set to True only if init succeeds
_mlflow = None  # the mlflow module, imported lazily


def _try_import_mlflow() -> bool:
    """Attempt to import mlflow. Returns True on success."""
    global _mlflow
    try:
        import mlflow as _mlflow_module

        _mlflow = _mlflow_module
        return True
    except ImportError:
        logger.warning("mlflow package not installed — tracking disabled.")
        return False


def init_mlflow() -> None:
    """
    Initialise MLflow tracking in a background thread so it never blocks startup.

    If the server is unreachable the thread will fail silently after one
    connection attempt (no long retry loop in the foreground).
    """
    if not _try_import_mlflow():
        return

    def _init():
        global _mlflow_enabled
        try:
            # Use a short socket timeout so we fail fast if server is down
            import mlflow

            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

            # set_experiment makes the first network call — wrap tightly
            mlflow.set_experiment(settings.mlflow_experiment_name)

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
        self._run = None
        self._start: float = 0.0

    def __enter__(self) -> "QueryTracker":
        self._start = time.perf_counter()
        if not _mlflow_enabled or _mlflow is None:
            return self
        try:
            self._run = _mlflow.start_run(
                run_name=f"query_{self.session_id[:8]}",
                tags={
                    "session_id": self.session_id,
                    "model": settings.model_name,
                    "component": "sql-agent",
                },
            )
            _mlflow.log_param("question", self.question[:500])
            _mlflow.log_param("session_id", self.session_id)
            _mlflow.log_param("model", settings.model_name)
        except Exception as exc:
            logger.debug(f"MLflow start_run skipped: {exc}")
            self._run = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        elapsed = round(time.perf_counter() - self._start, 3)
        if not self._run:
            return
        try:
            _mlflow.log_metric("latency_seconds", elapsed)
            if exc_type:
                _mlflow.set_tag("status", "error")
                _mlflow.log_param("error", str(exc_val)[:500])
                _mlflow.end_run(status="FAILED")
            else:
                _mlflow.set_tag("status", "success")
                _mlflow.end_run(status="FINISHED")
        except Exception as exc:
            logger.debug(f"MLflow end_run skipped: {exc}")

    def log_result(self, result: dict) -> None:
        """Log answer metrics — no-op if MLflow is disabled."""
        if not self._run:
            return
        try:
            answer = result.get("answer", "")
            chart_request = result.get("chart_request")
            messages = result.get("raw_messages", [])

            _mlflow.log_metric("answer_length_chars", len(answer))
            _mlflow.log_metric("message_count", len(messages))
            _mlflow.log_metric("chart_generated", 1 if chart_request else 0)
            if answer:
                _mlflow.log_param("answer_preview", answer[:300])
            if chart_request:
                _mlflow.log_param("chart_request", chart_request[:300])
        except Exception as exc:
            logger.debug(f"MLflow log_result skipped: {exc}")

    def log_chart(self, chart_result: dict) -> None:
        """Log chart metrics — no-op if MLflow is disabled."""
        if not self._run:
            return
        try:
            _mlflow.log_metric("chart_rows", chart_result.get("rows", 0))
            _mlflow.log_param("chart_type", chart_result.get("chart_type", "unknown"))
            _mlflow.set_tag("chart_title", chart_result.get("title", "")[:200])
        except Exception as exc:
            logger.debug(f"MLflow log_chart skipped: {exc}")


def track_query(session_id: str, question: str) -> QueryTracker:
    """Return a QueryTracker context manager for one agent turn."""
    return QueryTracker(session_id=session_id, question=question)
