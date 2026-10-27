"""
evals/scoring.py - Scoring rules for the evaluation set

Pure functions, no database or LLM access:
    compare_results(reference, agent, match)  -> (passed, reason)
    is_clarification(answer, sql)             -> bool
    score_question(spec, record, ref, agent)  -> dict
    summarize(scores, records)                -> dict
"""

import itertools
import math
import statistics
from collections import Counter
from typing import Any, Optional

CLARIFICATION_WORDS = ("specify", "clarify", "which", "could you", "please provide")


def _normalize(value: Any) -> Any:
    """Compare numbers to 2 decimals and text case-insensitively."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    return str(value).strip().casefold()


def _column(rows: list, j: int) -> list:
    return [_normalize(r[j]) for r in rows]


def _candidate_mappings(ref_rows: list, agent_rows: list, match: str) -> list[tuple]:
    """
    Possible assignments of each reference column to a distinct agent column
    whose values agree (whole column for set/ordered, first row for top_row).
    """
    n_ref = len(ref_rows[0])
    n_agent = len(agent_rows[0])
    candidates = []
    for j in range(n_ref):
        if match == "top_row":
            want = _normalize(ref_rows[0][j])
            cols = [k for k in range(n_agent) if _normalize(agent_rows[0][k]) == want]
        else:
            want = Counter(_column(ref_rows, j))
            cols = [k for k in range(n_agent) if Counter(_column(agent_rows, k)) == want]
        if not cols:
            return []
        candidates.append(cols)
    return [m for m in itertools.product(*candidates) if len(set(m)) == len(m)]


def compare_results(reference: list, agent: list, match: str = "set") -> tuple[bool, str]:
    """
    Execution-accuracy check: does the agent's result contain the reference result?

    Extra columns in the agent result are allowed (e.g. an id column); column
    names are ignored. `match` is "set", "ordered" or "top_row".
    """
    if match not in ("set", "ordered", "top_row"):
        raise ValueError(f"unknown match mode: {match!r}")
    if not reference:
        return (not agent, "both empty" if not agent else "reference is empty")
    if not agent:
        return False, "agent returned no rows"
    if match != "top_row" and len(agent) != len(reference):
        return False, f"row count {len(agent)} != expected {len(reference)}"

    for mapping in _candidate_mappings(reference, agent, match):
        projected = [tuple(_normalize(row[k]) for k in mapping) for row in agent]
        expected = [tuple(_normalize(v) for v in row) for row in reference]
        if match == "set" and Counter(projected) == Counter(expected):
            return True, "match"
        if match == "ordered" and projected == expected:
            return True, "match"
        if match == "top_row" and projected[0] == expected[0]:
            return True, "match"
    return False, "values do not match the reference"


def is_clarification(answer: str, sql: str) -> bool:
    """The agent asked the user a question instead of running SQL."""
    text = (answer or "").casefold()
    return not (sql or "").strip() and (
        "?" in text or any(w in text for w in CLARIFICATION_WORDS)
    )


def score_question(
    spec: dict,
    record: dict,
    reference_rows: Optional[list],
    agent_rows: Optional[list],
) -> dict:
    """Score one question. Checks that do not apply are reported as None."""
    result = {"id": spec["id"], "category": spec.get("category"), "checks": {}}
    checks = result["checks"]
    notes = []

    if record.get("error"):
        checks["no_error"] = False
        notes.append(f"error: {record['error']}")

    if spec.get("expects_clarification"):
        ok = is_clarification(record.get("answer", ""), record.get("sql", ""))
        checks["clarification"] = ok
        if not ok:
            notes.append("expected a clarifying question")
    elif spec.get("reference_sql"):
        ok, reason = compare_results(
            reference_rows or [], agent_rows or [], spec.get("match", "set")
        )
        checks["execution"] = ok
        if not ok:
            notes.append(reason)

    expects_chart = spec.get("expects_chart")
    if expects_chart is not None:
        chart_type = record.get("chart_type")
        if expects_chart:
            ok = chart_type is not None and chart_type == spec.get(
                "expected_chart_type", chart_type
            )
        else:
            ok = chart_type is None
        checks["chart"] = ok
        if not ok:
            notes.append(f"chart={chart_type!r}, expected {expects_chart!r}")

    result["passed"] = all(v for v in checks.values())
    result["notes"] = notes
    return result


def _rate(scores: list[dict], check: str) -> Optional[float]:
    values = [s["checks"][check] for s in scores if check in s["checks"]]
    return round(sum(values) / len(values), 3) if values else None


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * 0.95
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(scores: list[dict], records: list[dict]) -> dict:
    """Aggregate accuracy plus operational metrics (when the records have them)."""
    summary = {
        "questions": len(scores),
        "passed": sum(s["passed"] for s in scores),
        "execution_accuracy": _rate(scores, "execution"),
        "clarification_accuracy": _rate(scores, "clarification"),
        "chart_accuracy": _rate(scores, "chart"),
        "errors": sum(1 for r in records if r.get("error")),
    }
    for key, name in [
        ("latency_seconds", "latency"),
        ("llm_calls", "llm_calls"),
        ("cost_usd", "cost_usd"),
    ]:
        values = [r[key] for r in records if r.get(key) is not None]
        if values:
            summary[f"avg_{name}"] = round(statistics.mean(values), 4)
            if key == "latency_seconds":
                summary["p95_latency"] = round(_p95(values), 2)
    return summary


def check_thresholds(summary: dict, thresholds: dict) -> list[str]:
    """Return a list of threshold violations (empty when everything passes)."""
    failures = []
    for key, limit in thresholds.items():
        if key.startswith("min_"):
            metric = key[4:]
            value = summary.get(metric)
            if value is not None and value < limit:
                failures.append(f"{metric} = {value} < {limit}")
        elif key.startswith("max_"):
            metric = key[4:]
            value = summary.get(metric)
            if value is not None and value > limit:
                failures.append(f"{metric} = {value} > {limit}")
    return failures
