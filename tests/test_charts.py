import base64
from concurrent.futures import ThreadPoolExecutor

import pytest

from charts.generator import render_chart

ROWS = [("Rock", 1297), ("Latin", 579), ("Metal", 374), ("Jazz", 130)]
COLUMNS = ["genre", "track_count"]


def config(chart_type: str) -> dict:
    return {
        "chart_type": chart_type,
        "x_col": "genre",
        "y_col": "track_count",
        "title": "Tracks per genre",
        "x_label": "Genre",
        "y_label": "Tracks",
    }


def is_png(b64: str) -> bool:
    return base64.b64decode(b64).startswith(b"\x89PNG")


@pytest.mark.parametrize("chart_type", ["bar", "line", "pie", "scatter", "histogram"])
def test_renders_every_chart_type(chart_type):
    out = render_chart(ROWS, COLUMNS, config(chart_type))
    assert out["chart_type"] == chart_type
    assert out["rows"] == len(ROWS)
    assert is_png(out["image_base64"])


def test_single_value_result():
    out = render_chart([(275,)], ["artist_count"], config("bar"))
    assert is_png(out["image_base64"])


def test_unknown_columns_fall_back_to_first_two():
    cfg = config("bar") | {"x_col": "nope", "y_col": "missing"}
    out = render_chart(ROWS, COLUMNS, cfg)
    assert (out["x_col"], out["y_col"]) == ("genre", "track_count")


def test_no_rows_returns_error():
    out = render_chart([], COLUMNS, config("bar"))
    assert out["image_base64"] is None
    assert out["error"]


def test_concurrent_renders():
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda t: render_chart(ROWS, COLUMNS, config(t)), ["bar", "pie"] * 8)
        )
    assert all(is_png(r["image_base64"]) for r in results)


def test_non_numeric_values_are_counted():
    rows = [("Exotic Liquids", "UK"), ("Specialty Biscuits", "UK"), ("Tokyo Traders", "Japan")]
    cfg = config("bar") | {"x_col": "company", "y_col": "country"}
    out = render_chart(rows, ["company", "country"], cfg)
    assert is_png(out["image_base64"])
    assert (out["x_col"], out["y_col"], out["rows"]) == ("country", "count", 2)


def test_date_labels_are_readable():
    from charts.generator import _format_labels
    import pandas as pd

    months = pd.Series(["1997-01-01T00:00:00+00:00", "1997-02-01T00:00:00+00:00"])
    assert list(_format_labels(months)) == ["Jan 1997", "Feb 1997"]
    days = pd.Series(["1997-01-05", "1997-01-06"])
    assert list(_format_labels(days)) == ["1997-01-05", "1997-01-06"]
    names = pd.Series(["Rock", "Jazz"])
    assert list(_format_labels(names)) == ["Rock", "Jazz"]
