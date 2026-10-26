"""
charts/generator.py - Pure chart renderer

Takes actual SQL result rows + LLM-decided chart config and renders a chart.
No SQL execution. No keyword matching. No heuristics.

Handles all data shapes:
  - Single row  (e.g. count=275)  → bar chart with one bar
  - Multi-row   (e.g. genre/count) → bar, line, pie, scatter, histogram
  - Any column count ≥ 1

Public API:
    render_chart(rows, columns, config) -> dict
"""

import io
import base64
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

logger = logging.getLogger(__name__)

# ── Theme ─────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#2a2a3e",
    "axes.edgecolor": "#555577",
    "axes.labelcolor": "#cdd6f4",
    "xtick.color": "#cdd6f4",
    "ytick.color": "#cdd6f4",
    "text.color": "#cdd6f4",
    "grid.color": "#313145",
    "grid.linewidth": 0.8,
}
PALETTE = [
    "#89b4fa",
    "#cba6f7",
    "#89dceb",
    "#a6e3a1",
    "#fab387",
    "#f38ba8",
    "#f9e2af",
    "#94e2d5",
]


def render_chart(rows: list, columns: list[str], config: dict) -> dict:
    """
    Render a chart from SQL result rows using LLM-decided config.

    Args:
        rows:    List of row tuples, e.g. [("Rock", 1297), ("Jazz", 386)]
                 Can be a single row: [(275,)]
        columns: Column names, e.g. ["genre", "track_count"]
        config:  Dict from chart_config node:
                 {chart_type, x_col, y_col, title, x_label, y_label}

    Returns:
        {"image_base64": str, "chart_type": str, "title": str, "rows": int}
        or {"image_base64": None, "error": str}
    """
    if not config:
        return {"image_base64": None, "error": "No chart config provided."}

    chart_type = config.get("chart_type", "bar").lower()
    x_col = config.get("x_col", "")
    y_col = config.get("y_col", "")
    title = config.get("title", "")
    x_label = config.get("x_label", x_col)
    y_label = config.get("y_label", y_col)

    # ── Build DataFrame ───────────────────────────────────────────────────────
    if not rows:
        return {"image_base64": None, "error": "No data rows to render."}

    try:
        # Normalise rows: wrap scalars in tuples
        norm_rows = []
        for r in rows:
            if isinstance(r, (list, tuple)):
                norm_rows.append(tuple(r))
            else:
                norm_rows.append((r,))

        # Align columns to actual row width
        n = len(norm_rows[0])
        cols = (columns or [])[:n]
        while len(cols) < n:
            cols.append(f"col_{len(cols)}")

        df = pd.DataFrame(norm_rows, columns=cols)
    except Exception as exc:
        return {"image_base64": None, "error": f"DataFrame error: {exc}"}

    # ── Resolve x_col / y_col against actual columns (case-insensitive) ───────
    def resolve_col(name: str, df: pd.DataFrame) -> str | None:
        if name in df.columns:
            return name
        for c in df.columns:
            if c.lower() == name.lower():
                return c
        return None

    x_col_resolved = resolve_col(x_col, df)
    y_col_resolved = resolve_col(y_col, df)

    # Fallback: use first col as x, second (or first) as y
    if not x_col_resolved:
        x_col_resolved = df.columns[0]
        logger.warning(f"x_col '{x_col}' not found, using '{x_col_resolved}'")
    if not y_col_resolved:
        y_col_resolved = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        logger.warning(f"y_col '{y_col}' not found, using '{y_col_resolved}'")

    x_data = df[x_col_resolved].astype(str)
    y_data = pd.to_numeric(df[y_col_resolved], errors="coerce").fillna(0)

    # ── Render ────────────────────────────────────────────────────────────────
    try:
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor("#1e1e2e")

            if chart_type == "pie":
                _pie(ax, x_data, y_data)
            elif chart_type == "line":
                _line(ax, x_data, y_data)
            elif chart_type == "scatter":
                x_num = pd.to_numeric(df[x_col_resolved], errors="coerce").fillna(0)
                _scatter(ax, x_num, y_data, x_data)
            elif chart_type == "histogram":
                _histogram(ax, y_data)
            else:
                _bar(ax, x_data, y_data)  # default: bar

            # Labels
            ax.set_title(title, fontsize=13, fontweight="bold", pad=15, color="#cdd6f4")
            if chart_type not in ("pie",):
                ax.set_xlabel(
                    (x_label or x_col_resolved).replace("_", " ").title(), fontsize=10
                )
                ax.set_ylabel(
                    (y_label or y_col_resolved).replace("_", " ").title(), fontsize=10
                )

            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(
                buf,
                format="png",
                dpi=130,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            plt.close(fig)
            buf.seek(0)

        return {
            "image_base64": base64.b64encode(buf.read()).decode("utf-8"),
            "chart_type": chart_type,
            "title": title,
            "x_col": x_col_resolved,
            "y_col": y_col_resolved,
            "rows": len(df),
        }

    except Exception as exc:
        plt.close("all")
        logger.error(f"render_chart error: {exc}", exc_info=True)
        return {"image_base64": None, "error": str(exc)}


# ── Chart type renderers ──────────────────────────────────────────────────────


def _bar(ax, x_data, y_data):
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(x_data))]
    bars = ax.bar(
        range(len(x_data)),
        y_data,
        color=colors,
        edgecolor="#1e1e2e",
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xticks(range(len(x_data)))
    ax.set_xticklabels(x_data, rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.1f}"))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for bar in bars:
        h = bar.get_height()
        if h != 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.01,
                f"{h:,.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#cdd6f4",
            )


def _line(ax, x_data, y_data):
    xs = range(len(x_data))
    ax.plot(
        xs,
        y_data,
        color=PALETTE[0],
        linewidth=2.5,
        marker="o",
        markersize=5,
        markerfacecolor=PALETTE[1],
    )
    ax.fill_between(xs, y_data, alpha=0.12, color=PALETTE[0])
    ax.set_xticks(xs)
    ax.set_xticklabels(x_data, rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.1f}"))
    ax.grid(axis="y", alpha=0.3)


def _pie(ax, x_data, y_data):
    wedges, texts, autotexts = ax.pie(
        y_data,
        labels=x_data,
        autopct="%1.1f%%",
        colors=PALETTE[: len(y_data)],
        startangle=140,
        pctdistance=0.85,
        wedgeprops={"width": 0.6},
    )
    for t in autotexts:
        t.set_color("#1e1e2e")
        t.set_fontsize(9)


def _scatter(ax, x_num, y_data, labels):
    ax.scatter(x_num, y_data, color=PALETTE[0], alpha=0.75, s=80, zorder=3)
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (x_num.iloc[i], y_data.iloc[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7.5,
            color="#cdd6f4",
            alpha=0.85,
        )
    ax.grid(alpha=0.3)


def _histogram(ax, y_data):
    bins = min(20, max(5, len(y_data) // 2))
    ax.hist(y_data, bins=bins, color=PALETTE[0], edgecolor="#1e1e2e")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
