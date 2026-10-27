import json
import pathlib

import pytest
import yaml

from evals.scoring import (
    check_thresholds,
    compare_results,
    is_clarification,
    score_question,
    summarize,
)

EVALS = pathlib.Path(__file__).parent.parent / "evals"

REF = [("Beverages", 267868.18), ("Dairy Products", 234507.28)]


def test_set_match_ignores_order_extra_columns_and_case():
    agent = [(4, "dairy products", 234507.2811), (1, "Beverages", 267868.18)]
    assert compare_results(REF, agent, "set") == (True, "match")


def test_ordered_match_requires_same_order():
    assert compare_results(REF, list(reversed(REF)), "ordered")[0] is False
    assert compare_results(REF, REF, "ordered")[0] is True


def test_top_row_allows_limit_1_or_all_groups():
    assert compare_results(REF, [("Beverages", 267868.18)], "top_row")[0] is True
    assert compare_results(REF, REF + [("Seafood", 1.0)], "top_row")[0] is True
    assert compare_results(REF, [("Seafood", 1.0)], "top_row")[0] is False


def test_truncated_result_fails():
    ref = [(f"p{i}",) for i in range(18)]
    ok, reason = compare_results(ref, ref[:10], "set")
    assert not ok and "row count 10" in reason


def test_wrong_values_fail():
    assert compare_results(REF, [("Beverages", 1.0), ("Dairy Products", 2.0)], "set")[0] is False


def test_empty_agent_result_fails():
    assert compare_results(REF, [], "set") == (False, "agent returned no rows")


def test_unknown_match_mode():
    with pytest.raises(ValueError):
        compare_results(REF, REF, "fuzzy")


@pytest.mark.parametrize(
    "answer, sql, expected",
    [
        ("Could you please specify which products you mean?", "", True),
        ("Please specify the products you are interested in.", "", True),
        ("There are 12 products.", "SELECT COUNT(*) FROM products", False),
        ("Which products? Here is a list anyway.", "SELECT 1", False),
    ],
)
def test_is_clarification(answer, sql, expected):
    assert is_clarification(answer, sql) is expected


def test_score_question_chart_checks():
    spec = {"id": "q", "reference_sql": "x", "match": "set", "expects_chart": True,
            "expected_chart_type": "pie"}
    ok = score_question(spec, {"chart_type": "pie"}, REF, REF)
    wrong_type = score_question(spec, {"chart_type": "bar"}, REF, REF)
    assert ok["passed"] and not wrong_type["passed"]
    assert wrong_type["checks"] == {"execution": True, "chart": False}


def test_optional_chart_is_not_scored():
    spec = {"id": "q", "reference_sql": "x", "match": "set", "expects_chart": None}
    assert "chart" not in score_question(spec, {"chart_type": None}, REF, REF)["checks"]


def test_summarize_and_thresholds():
    scores = [
        {"passed": True, "checks": {"execution": True, "chart": True}},
        {"passed": False, "checks": {"execution": False}},
    ]
    records = [{"latency_seconds": 4.0, "llm_calls": 3, "cost_usd": 0.01},
               {"latency_seconds": 6.0, "llm_calls": 5, "cost_usd": 0.02}]
    summary = summarize(scores, records)
    assert summary["execution_accuracy"] == 0.5
    assert summary["avg_llm_calls"] == 4
    failures = check_thresholds(summary, {"min_execution_accuracy": 0.85, "max_avg_llm_calls": 4.0})
    assert failures == ["execution_accuracy = 0.5 < 0.85"]


def test_eval_files_are_consistent():
    questions = yaml.safe_load((EVALS / "questions.yaml").read_text(encoding="utf-8"))["questions"]
    run = json.loads((EVALS / "results" / "optimized_run.json").read_text(encoding="utf-8"))
    baseline = json.loads((EVALS / "baseline.json").read_text(encoding="utf-8"))

    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    assert ids == [q["id"] for q in run["questions"]]
    for q in questions:
        assert q.get("reference_sql") or q.get("expects_clarification"), q["id"]
        assert q.get("match", "set") in ("set", "ordered", "top_row")
    assert check_thresholds(baseline["recorded"], baseline["thresholds"]) == []
