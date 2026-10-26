"""
app/agent.py - LangGraph SQL Agent

Graph flow:
    START
      │
      ▼
  [query_rewriter]     ← resolves follow-up references using history
      │
      ▼
  [intent_detector]    ← classifies if user wants a chart (explicit or implied)
      │                        sets state["wants_chart"] = True/False
      ▼
  [generate_query]     ← schema is in the system prompt (loaded once at startup);
      │                  writes SQL query, stores it in state["last_sql"]
      │
      ├─(tool call)──► [check_query] ──► [run_query] ──► back to [generate_query]
      │
      └─(final answer)
            │
            ├── wants_chart=True  ──► [chart_config] ──► END
            │                         (LLM decides chart type/axes from actual rows)
            └── wants_chart=False ──► END

State:
    messages      - conversation messages
    wants_chart   - bool: did the user ask for a chart (explicit or implied)?
    last_sql      - the last SQL query executed (shown in response)
    sql_rows      - raw result rows from last run_query
    sql_columns   - column names for sql_rows
    chart_config  - LLM-decided chart config dict, or None
"""

import json
import logging
import re
from typing import Annotated, Any, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from sqlalchemy import create_engine
from typing_extensions import TypedDict

from app.config import get_settings
from app.memory import get_history_as_langchain_messages, store_message
from app.sql_runner import execute_readonly

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_FETCH_ROWS = 100  # hard cap on rows fetched per query

# ══════════════════════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════════════════════


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    wants_chart: bool  # set by intent_detector
    last_sql: str  # last SQL query executed
    sql_rows: list  # raw result rows
    sql_columns: list[str]  # column names
    chart_config: Optional[dict]  # decided by chart_config node


# ══════════════════════════════════════════════════════════════════════════════
# build_agent
# ══════════════════════════════════════════════════════════════════════════════


def build_agent() -> tuple[Any, SQLDatabase]:

    # ── Models ────────────────────────────────────────────────────────────────
    main_model = init_chat_model(
        settings.model_name, model_provider="openai", temperature=0
    )
    chart_model = init_chat_model(
        "gpt-4.1-mini", model_provider="openai", temperature=0
    )

    # ── DB ────────────────────────────────────────────────────────────────────
    engine = create_engine(
        settings.business_db_url,
        pool_pre_ping=True,
        connect_args={
            "options": f"-c statement_timeout={settings.query_timeout_seconds * 1000}"
        },
    )
    db = SQLDatabase(engine, sample_rows_in_table_info=3)
    tables = db.get_usable_table_names()
    logger.info(f"DB: dialect={db.dialect} tables={tables}")

    # Schema (CREATE TABLE statements + sample rows) is loaded once at startup
    # and embedded in the system prompt. Restart the app after schema changes.
    schema_info = db.get_table_info()
    logger.info(f"Schema loaded: {len(tables)} tables, {len(schema_info)} chars")

    # ── Tools ─────────────────────────────────────────────────────────────────
    toolkit = SQLDatabaseToolkit(db=db, llm=main_model)
    run_query_tool = next(t for t in toolkit.get_tools() if t.name == "sql_db_query")

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 1 — query_rewriter
    # Rewrites follow-up questions to be self-contained using history.
    # ══════════════════════════════════════════════════════════════════════════
    REWRITER_SYSTEM = """You are a query-rewriting assistant for a conversational BI system.

Rewrite the user's latest question into a fully self-contained, unambiguous question.

Rules:
- If already self-contained, output it unchanged.
- Resolve pronouns (it, they, that, those) using history.
- Expand follow-ups ("what about X?", "and for Y?") into full standalone questions.
- Replace vague references ("the top one", "that genre") with actual values from history.
- Never answer the question — only rewrite it.
- Output ONLY the rewritten question. No preamble.
"""

    def query_rewriter(state: AgentState) -> dict:
        messages = state["messages"]
        last_human_idx = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if getattr(messages[i], "type", None) == "human"
            ),
            None,
        )
        if last_human_idx is None:
            return {}

        original = messages[last_human_idx].content
        has_history = any(
            getattr(m, "type", None) in ("human", "ai")
            for m in messages[:last_human_idx]
        )
        if not has_history:
            return {}

        prompt = [{"role": "system", "content": REWRITER_SYSTEM}] + [
            {
                "role": "user" if getattr(m, "type", None) == "human" else "assistant",
                "content": m.content if isinstance(m.content, str) else str(m.content),
            }
            for m in messages
            if getattr(m, "type", None) in ("human", "ai")
            and isinstance(m.content, str)
            and m.content.strip()
        ]
        rewritten = main_model.invoke(prompt).content.strip()
        if rewritten and rewritten != original:
            logger.info(f"rewriter: '{original}' → '{rewritten}'")
            updated = list(messages)
            updated[last_human_idx] = HumanMessage(content=rewritten)
            return {"messages": updated}
        return {}

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 2 — intent_detector
    #
    # Determines wants_chart from the user's latest message.
    # Two triggers:
    #   A) EXPLICIT  — user says "chart", "graph", "plot", "visualise", etc.
    #   B) IMPLIED   — user asks a comparative/trend/distribution question where
    #                  a chart would naturally help, even without saying "chart"
    #
    # Uses gpt-4.1-mini with a strict yes/no prompt — fast and cheap.
    # ══════════════════════════════════════════════════════════════════════════
    INTENT_SYSTEM = """You are a chart-intent classifier for a BI assistant.

Given the user's question, decide if they want a chart/graph/visualisation.

Return ONLY one word: YES or NO

Classify as YES if:
- The user explicitly mentions: chart, graph, plot, visualise, visualize, bar, pie,
  line chart, histogram, scatter, diagram, show me a graph, draw, display as chart
- The user asks for a comparison, trend, distribution, ranking, or breakdown where
  a chart would clearly be the most useful output

Classify as NO if:
- The user asks a simple factual question (how many, what is, who, when, list)
- The user wants a text answer or table, not a visual
- The question is conversational or asks for an explanation

Respond with ONLY: YES or NO
"""

    def intent_detector(state: AgentState) -> dict:
        messages = state["messages"]
        # Get the current user question (last human message)
        user_q = ""
        for m in reversed(messages):
            if getattr(m, "type", None) == "human":
                user_q = m.content if isinstance(m.content, str) else ""
                break

        if not user_q:
            return {"wants_chart": False}

        try:
            resp = main_model.invoke(
                [
                    {"role": "system", "content": INTENT_SYSTEM},
                    {"role": "user", "content": user_q},
                ]
            )
            answer = resp.content.strip().upper()
            wants = answer.startswith("YES")
            logger.info(f"intent_detector: wants_chart={wants} (raw='{answer[:20]}')")
            return {"wants_chart": wants}
        except Exception as exc:
            logger.warning(f"intent_detector failed, defaulting to False: {exc}")
            return {"wants_chart": False}

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 3 — generate_query
    #
    # The database schema is part of the system prompt, so no schema
    # lookup step is needed. The static part of the prompt comes first so
    # the provider's prompt cache can reuse it across requests.
    # The system prompt is dynamically adjusted based on wants_chart.
    # When chart is wanted: instructs the LLM to write a query suited for
    # visualisation (multi-column, grouped where useful) AND append CHART_REQUEST.
    # When no chart: standard answer prompt.
    # ══════════════════════════════════════════════════════════════════════════
    BASE_GENERATE_SYSTEM = f"""You are an expert SQL analyst and BI assistant for a {db.dialect} database.

Database schema (tables, columns and sample rows):
{schema_info}

Your job:
1. Write a correct, efficient SQL query that answers the user's question.
2. Execute it and interpret the results clearly.
3. Give a clean, user-facing natural language answer.

Rules:
- NEVER show SQL query text in your answer — the system surfaces it separately.
- NEVER include code blocks, markdown, or raw SQL in the answer text.
- Write only what a business user wants to read.
- Limit results to at most {settings.max_results} rows unless user specifies.
- Order by the most relevant column.
- NEVER run DML statements (INSERT, UPDATE, DELETE, DROP, TRUNCATE).
- If the question cannot be answered from the available data, say so clearly.
"""

    CHART_GENERATE_ADDITION = """
CHART MODE — the user wants a visualisation:
- Write a query that returns data SUITABLE FOR CHARTING.
- The query MUST return at least 2 columns: one label/category and one numeric value.
- If the user asked a scalar question (e.g. "how many artists total"):
    → Do NOT just return COUNT(*). Instead return a meaningful breakdown, e.g.:
       artists grouped by number of albums, genres by track count, etc.
- Prefer GROUP BY queries that produce multiple rows for interesting visual breakdowns.
- After your text answer, append exactly this token on its own line: CHART_REQUEST
"""

    def generate_query(state: AgentState) -> dict:
        wants = state.get("wants_chart", False)
        system_content = BASE_GENERATE_SYSTEM + (
            CHART_GENERATE_ADDITION if wants else ""
        )
        system = {"role": "system", "content": system_content}
        llm = main_model.bind_tools([run_query_tool])
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 4 — check_query
    # ══════════════════════════════════════════════════════════════════════════
    CHECK_SYSTEM = f"""You are a {db.dialect} SQL expert. Double-check the query for:
- NOT IN with NULLs, UNION vs UNION ALL, exclusive BETWEEN ranges
- Type mismatches, wrong quoting, wrong argument counts, incorrect casts, wrong join columns

Rewrite only if there are mistakes. Otherwise reproduce unchanged.
Always call sql_db_query to execute it.
"""

    def check_query(state: AgentState) -> dict:
        tool_call = state["messages"][-1].tool_calls[0]
        llm = main_model.bind_tools([run_query_tool], tool_choice="any")
        response = llm.invoke(
            [
                {"role": "system", "content": CHECK_SYSTEM},
                {"role": "user", "content": tool_call["args"]["query"]},
            ]
        )
        response.id = state["messages"][-1].id
        return {"messages": [response]}

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 5 — run_query  (custom — captures rows + SQL into state)
    # ══════════════════════════════════════════════════════════════════════════
    def run_query(state: AgentState) -> dict:
        last_msg = state["messages"][-1]
        tool_call = last_msg.tool_calls[0]
        sql = tool_call["args"].get("query", "")

        result = execute_readonly(engine, sql, max_rows=MAX_FETCH_ROWS)

        logger.info(
            f"run_query: {len(result.rows)} rows | columns={result.columns} "
            f"| error={result.error is not None} | sql={sql[:80]}"
        )

        tool_msg = ToolMessage(
            content=result.to_tool_content(),
            tool_call_id=tool_call["id"],
            name=run_query_tool.name,
        )
        return {
            "messages": [tool_msg],
            "last_sql": sql,
            "sql_rows": result.rows,
            "sql_columns": result.columns,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 6 — chart_config
    #
    # Receives actual sql_rows + sql_columns from state.
    # Asks gpt-4.1-mini to decide the best chart type and column mapping.
    # Always produces a chart config — NEVER returns "none" here because
    # wants_chart=True is the only reason this node runs.
    # If data is truly unplottable, it will fabricate the best possible config
    # (e.g. a single-value bar chart is still valid).
    # ══════════════════════════════════════════════════════════════════════════
    CHART_CONFIG_SYSTEM = """You are a data visualisation expert for a BI system.

You will receive: the user's question, column names, sample rows, and the agent's text answer.

Your job: decide the BEST chart type and column mapping for this data.

Chart types available: bar, line, pie, scatter, histogram

Guidelines:
- bar:       Best for comparisons between categories. Use when x is categorical, y is numeric.
- line:      Best for trends over time. Use when x is ordered/temporal, y is numeric.
- pie:       Best for proportions. Use only when there are ≤ 12 categories and they sum to a whole.
- scatter:   Best for correlation between two numeric variables.
- histogram: Best for showing distribution of a single numeric column.

Special case — single row / scalar result:
- If there is only 1 row (e.g. total count = 275), use a bar chart with the label as x and value as y.
- Never refuse to produce a chart config.

x_col and y_col MUST be exact column names from the provided columns list.

Return ONLY valid JSON, no markdown fences, no explanation:
{
  "chart_type": "<bar|line|pie|scatter|histogram>",
  "x_col":      "<exact column name>",
  "y_col":      "<exact column name>",
  "title":      "<concise descriptive title>",
  "x_label":    "<x-axis label>",
  "y_label":    "<y-axis label>"
}
"""

    def chart_config(state: AgentState) -> dict:
        rows = state.get("sql_rows", [])
        columns = state.get("sql_columns", [])
        sql = state.get("last_sql", "")

        # Get user question and agent answer from messages
        user_q, agent_ans = "", ""
        for m in reversed(state["messages"]):
            t = getattr(m, "type", "")
            c = getattr(m, "content", "") or ""
            if t == "ai" and c and not getattr(m, "tool_calls", None) and not agent_ans:
                agent_ans = (
                    (c if isinstance(c, str) else str(c))
                    .replace("CHART_REQUEST", "")
                    .strip()
                )
            if t == "human" and not user_q:
                user_q = c if isinstance(c, str) else str(c)

        sample = rows[:15]
        prompt = f"""User question: {user_q}

SQL executed: {sql}

Columns: {columns}
Rows ({len(rows)} total, showing up to 15): {sample}

Agent answer: {agent_ans}

Decide the best chart config and return JSON."""

        try:
            raw = chart_model.invoke(
                [
                    {"role": "system", "content": CHART_CONFIG_SYSTEM},
                    {"role": "user", "content": prompt},
                ]
            ).content.strip()

            # Strip code fences if model adds them
            raw = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE
            ).strip()
            cfg = json.loads(raw)
            logger.info(f"chart_config: {cfg}")
            return {"chart_config": cfg}

        except Exception as exc:
            logger.warning(f"chart_config LLM failed: {exc}")
            # Fallback: build a minimal valid config from whatever we have
            x = columns[0] if columns else "label"
            y = columns[1] if len(columns) > 1 else columns[0] if columns else "value"
            return {
                "chart_config": {
                    "chart_type": "bar",
                    "x_col": x,
                    "y_col": y,
                    "title": user_q[:60],
                    "x_label": x,
                    "y_label": y,
                }
            }

    # ══════════════════════════════════════════════════════════════════════════
    # Routing
    # ══════════════════════════════════════════════════════════════════════════

    def after_generate(
        state: AgentState,
    ) -> Literal["check_query", "chart_config", "__end__"]:
        last = state["messages"][-1]
        # Still has a tool call → keep running SQL
        if getattr(last, "tool_calls", None):
            return "check_query"
        # Final answer — route based on wants_chart flag (set by intent_detector)
        if state.get("wants_chart", False):
            return "chart_config"
        return END

    # ══════════════════════════════════════════════════════════════════════════
    # Assemble graph
    # ══════════════════════════════════════════════════════════════════════════
    builder = StateGraph(AgentState)

    builder.add_node("query_rewriter", query_rewriter)
    builder.add_node("intent_detector", intent_detector)
    builder.add_node("generate_query", generate_query)
    builder.add_node("check_query", check_query)
    builder.add_node("run_query", run_query)
    builder.add_node("chart_config", chart_config)

    builder.add_edge(START, "query_rewriter")
    builder.add_edge("query_rewriter", "intent_detector")
    builder.add_edge("intent_detector", "generate_query")
    builder.add_conditional_edges("generate_query", after_generate)
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")
    builder.add_edge("chart_config", END)

    agent = builder.compile()
    logger.info(
        "Agent compiled: rewriter→intent→generate⇄[check→run]→[chart]→END"
    )
    return agent, db


# ══════════════════════════════════════════════════════════════════════════════
# run_agent
# ══════════════════════════════════════════════════════════════════════════════


async def run_agent(agent: Any, question: str, session_id: str) -> dict:
    """
    Run one agent turn. Returns answer, sql, chart_config, and metadata.
    """
    history = get_history_as_langchain_messages(session_id, limit=20)
    logger.info(f"[{session_id}] {len(history)} history msgs loaded.")

    initial: AgentState = {
        "messages": history + [HumanMessage(content=question)],
        "wants_chart": False,
        "last_sql": "",
        "sql_rows": [],
        "sql_columns": [],
        "chart_config": None,
    }

    final: AgentState = await agent.ainvoke(initial)

    messages = final.get("messages", [])
    sql_rows = final.get("sql_rows", [])
    sql_columns = final.get("sql_columns", [])
    last_sql = final.get("last_sql", "")
    chart_cfg = final.get("chart_config", None)
    wants_chart = final.get("wants_chart", False)

    # Extract clean final answer (strip CHART_REQUEST marker if present)
    final_answer = ""
    for msg in reversed(messages):
        t = getattr(msg, "type", "")
        c = getattr(msg, "content", "")
        if t == "ai" and c and not getattr(msg, "tool_calls", None):
            final_answer = c if isinstance(c, str) else str(c)
            final_answer = final_answer.replace("CHART_REQUEST", "").strip()
            break

    # Extract rewritten question
    rewritten_q = question
    for msg in messages[len(history) :]:
        if getattr(msg, "type", None) == "human":
            rewritten_q = msg.content if isinstance(msg.content, str) else question
            break

    # Persist to memory DB
    if question.strip():
        store_message(session_id, role="user", content=question)
    if final_answer.strip():
        store_message(session_id, role="assistant", content=final_answer)

    logger.info(
        f"[{session_id}] done | wants_chart={wants_chart} "
        f"rows={len(sql_rows)} chart={'yes' if chart_cfg else 'no'} "
        f"sql='{last_sql[:60]}'"
    )

    return {
        "answer": final_answer,
        "sql": last_sql,  # shown in API response
        "chart_config": chart_cfg,
        "wants_chart": wants_chart,
        "rewritten_q": rewritten_q,
        "sql_rows": sql_rows,
        "sql_columns": sql_columns,
        "raw_messages": messages,
    }
