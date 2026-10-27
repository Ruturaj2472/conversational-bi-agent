"""
evals/run_eval.py - Score the agent against evals/questions.yaml

Modes:
    --replay FILE   Score a recorded run (no LLM calls). Default:
                    evals/results/optimized_run.json
    --live          Send every question to a running API and score the answers
                    (calls OpenAI through the agent).

Both modes execute the reference SQL and the agent's SQL against PostgreSQL
(read-only) and compare the results.

Examples:
    python -m evals.run_eval                                # replay the recorded run
    python -m evals.run_eval --check                        # ...and fail on threshold violations
    python -m evals.run_eval --live --api http://localhost:8000 --save evals/results/live.json

Database connection: EVAL_DB_URL, or the POSTGRES_* environment variables.
"""

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
import uuid

import yaml
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from app.sql_runner import execute_readonly
from evals.scoring import check_thresholds, score_question, summarize

EVALS_DIR = pathlib.Path(__file__).parent
DEFAULT_QUESTIONS = EVALS_DIR / "questions.yaml"
DEFAULT_RUN = EVALS_DIR / "results" / "optimized_run.json"
DEFAULT_BASELINE = EVALS_DIR / "baseline.json"


def db_url() -> str:
    if os.getenv("EVAL_DB_URL"):
        return os.environ["EVAL_DB_URL"]
    return URL.create(
        "postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "northwind"),
    ).render_as_string(hide_password=False)


def load_questions(path: pathlib.Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def _chart_type(chart) -> str | None:
    """Recorded runs store e.g. "pie (16 rows)"; live responses store a dict."""
    if not chart:
        return None
    if isinstance(chart, dict):
        return chart.get("chart_type")
    return str(chart).split()[0]


def load_replay(path: pathlib.Path) -> dict[str, dict]:
    run = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    for q in run["questions"]:
        records[q["id"]] = {**q, "chart_type": _chart_type(q.get("chart"))}
    return records


def run_live(questions: list[dict], api: str) -> dict[str, dict]:
    """Ask every question through POST /chart; one session per group."""
    sessions = {}
    records = {}
    for spec in questions:
        session = sessions.setdefault(spec["group"], f"eval-{uuid.uuid4().hex[:6]}")
        body = json.dumps({"message": spec["question"], "session_id": session}).encode()
        req = urllib.request.Request(
            f"{api.rstrip('/')}/chart",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
            error = None
        except (urllib.error.URLError, TimeoutError) as exc:
            data, error = {}, str(exc)
        records[spec["id"]] = {
            "id": spec["id"],
            "question": spec["question"],
            "answer": data.get("answer", ""),
            "sql": data.get("sql", ""),
            "wants_chart": data.get("wants_chart"),
            "chart_type": _chart_type(data.get("chart")),
            "latency_seconds": round(time.perf_counter() - t0, 2),
            "error": error,
        }
        print(f"  {spec['id']} {records[spec['id']]['latency_seconds']:5.1f}s", flush=True)
    return records


def evaluate(questions: list[dict], records: dict[str, dict], engine) -> tuple[list, dict]:
    scores = []
    for spec in questions:
        record = records.get(spec["id"], {"error": "question missing from run"})
        ref_rows = agent_rows = None
        if spec.get("reference_sql") and not spec.get("expects_clarification"):
            ref = execute_readonly(engine, spec["reference_sql"], max_rows=1000)
            if ref.error:
                raise RuntimeError(f"{spec['id']}: reference SQL failed: {ref.error}")
            ref_rows = ref.rows
            if record.get("sql"):
                agent = execute_readonly(engine, record["sql"], max_rows=1000)
                agent_rows = [] if agent.error else agent.rows
        scores.append(score_question(spec, record, ref_rows, agent_rows))
    summary = summarize(scores, [records.get(q["id"], {}) for q in questions])
    return scores, summary


def print_report(scores: list[dict], summary: dict) -> None:
    print(f"\n{'id':<5} {'category':<13} {'result':<6} checks")
    for s in scores:
        checks = " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in s["checks"].items())
        notes = f"  ({'; '.join(s['notes'])})" if s["notes"] else ""
        print(f"{s['id']:<5} {s['category'] or '-':<13} {'PASS' if s['passed'] else 'FAIL':<6} {checks}{notes}")
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key:<24} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=pathlib.Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--replay", type=pathlib.Path, default=DEFAULT_RUN)
    parser.add_argument("--live", action="store_true", help="query a running API")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--save", type=pathlib.Path, help="save live answers as a run file")
    parser.add_argument("--baseline", type=pathlib.Path, default=DEFAULT_BASELINE)
    parser.add_argument("--check", action="store_true", help="exit 1 on threshold violations")
    parser.add_argument("--report", type=pathlib.Path, help="write the full report as JSON")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.live:
        print(f"Live run against {args.api}")
        records = run_live(questions, args.api)
        if args.save:
            payload = {"questions": list(records.values())}
            args.save.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        print(f"Replaying {args.replay}")
        records = load_replay(args.replay)

    engine = create_engine(db_url(), pool_pre_ping=True)
    scores, summary = evaluate(questions, records, engine)
    print_report(scores, summary)

    if args.report:
        args.report.write_text(
            json.dumps({"summary": summary, "scores": scores}, indent=2) + "\n", encoding="utf-8"
        )

    if args.check:
        thresholds = json.loads(args.baseline.read_text(encoding="utf-8"))["thresholds"]
        failures = check_thresholds(summary, thresholds)
        if failures:
            print("\nThreshold violations:")
            for f in failures:
                print(f"  ✗ {f}")
            return 1
        print("\nAll thresholds met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
