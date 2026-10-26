"""
app/agent.py - LangGraph SQL Agent

Graph flow:
    START
      │
      ▼
  [analyze_question]   ← one fast-model call: resolves follow-up references
      │                  using history AND classifies if the user wants a chart
      │                  (explicit or implied), sets state["wants_chart"]
      ▼
  [generate_query]     ← schema is in the system prompt (loaded once at startup);
      │                  writes SQL query, stores it in state["last_sql"]
      │
      ├─(tool call)──► [run_query] ──► back to [generate_query]
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

import logging
from typing import Annotated, Any, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from typing_extensions import TypedDict

from app.config import get_settings
from app.memory import get_history_as_langchain_messages, store_message
from app.sql_runner import execute_readonly

logger = logging.getLogger(__name__)
settings = get_settings()

MAX_FETCH_ROWS = 100  # hard cap on rows fetched per query
RECURSION_LIMIT = 12  # graph steps per turn; allows ~4 SQL attempts
GIVE_UP_ANSWER = (
    "I couldn't build a working query for that question. "
    "Please try rephrasing it or making it more specific."
)

# ══════════════════════════════════════════════════════════════════════════════
# State
# ══════════════════════════════════════════════════════════════════════════════


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    wants_chart: bool  # set by analyze_question
    last_sql: str  # last SQL query executed
    sql_rows: list  # raw result rows
    sql_columns: list[str]  # column names
    chart_config: Optional[dict]  # decided by chart_config node


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _current_turn(messages: list) -> list:
    """
    Messages from the latest human question onward.

    analyze_question has already rewritten that question to be self-contained,
    so SQL generation does not need the earlier conversation.
    """
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            return messages[i:]
    return messages


# ══════════════════════════════════════════════════════════════════════════════
# build_agent
# ══════════════════════════════════════════════════════════════════════════════


def build_agent() -> tuple[Any, SQLDatabase]:

    # ── Models ────────────────────────────────────────────────────────────────
    main_model = init_chat_model(
        settings.model_name, model_provider="openai", temperature=0
    )
    fast_model = init_chat_model(
        settings.fast_model_name, model_provider="openai", temperature=0
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
    # NODE 1 — analyze_question
    #
    # One structured-output call on the fast model that both:
    #   - rewrites follow-up questions to be self-contained using history
    #   - decides wants_chart, either
    #       A) EXPLICIT — user says "chart", "graph", "plot", "visualise", etc.
    #       B) IMPLIED  — a comparison/trend/distribution question where a
    #                     chart would naturally help
    # ══════════════════════════════════════════════════════════════════════════
    ANALYZE_SYSTEM = """You analyse the user's latest question for a conversational BI system.

Return two things:

1. rewritten_question — the latest question as a fully self-contained, unambiguous question.
- If already self-contained, return it unchanged.
- Resolve pronouns (it, they, that, those) using the conversation history.
- Expand follow-ups ("what about X?", "and for Y?") into full standalone questions.
- Replace vague references ("the top one", "that genre") with actual values from history.
- Never answer the question — only rewrite it.

2. wants_chart — true if the user wants a chart/graph/visualisation:
- true if the user explicitly mentions: chart, graph, plot, visualise, visualize, bar, pie,
  line chart, histogram, scatter, diagram, show me a graph, draw, display as chart
- true if the user asks for a comparison, trend, distribution, ranking, or breakdown where
  a chart would clearly be the most useful output
- false for simple factual questions (how many, what is, who, when, list)
- false if the user wants a text answer or table, or the question is conversational
"""

    class QuestionAnalysis(BaseModel):
        rewritten_question: str = Field(
            description="The latest question rewritten to be self-contained."
        )
        wants_chart: bool = Field(
            description="Whether a chart should be produced for this question."
        )

    analyzer = fast_model.with_structured_output(QuestionAnalysis)

    def analyze_question(state: AgentState) -> dict:
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
            return {"wants_chart": False}

        last_human = messages[last_human_idx]
        original = last_human.content if isinstance(last_human.content, str) else ""
        history_lines = [
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
            for m in messages[:last_human_idx]
            if getattr(m, "type", None) in ("human", "ai")
            and isinstance(m.content, str)
            and m.content.strip()
        ]
        history_text = "\n".join(history_lines) or "(no previous messages)"

        try:
            result = analyzer.invoke(
                [
                    {"role": "system", "content": ANALYZE_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Conversation history:\n{history_text}\n\n"
                        f"Latest question: {original}",
                    },
                ]
            )
        except Exception as exc:
            logger.warning(f"analyze_question failed, using original question: {exc}")
            return {"wants_chart": False}

        rewritten = (result.rewritten_question or "").strip() or original
        logger.info(
            f"analyze_question: wants_chart={result.wants_chart} "
            f"'{original}' → '{rewritten}'"
        )
        update: dict = {"wants_chart": result.wants_chart}
        if rewritten != original:
            # Same id → add_messages replaces the original question in place
            update["messages"] = [HumanMessage(content=rewritten, id=last_human.id)]
        return update

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 2 — generate_query
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
- Before running a query, check it for common mistakes: NOT IN with NULLs,
  UNION vs UNION ALL, exclusive BETWEEN ranges, type mismatches, wrong quoting,
  wrong argument counts, incorrect casts, and wrong join columns.
- Run one query at a time. If a query returns an error, fix it and try again.
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
        llm = main_model.bind_tools([run_query_tool], parallel_tool_calls=False)
        response = llm.invoke([system] + _current_turn(state["messages"]))
        return {"messages": [response]}

    # ══════════════════════════════════════════════════════════════════════════
    # NODE 3 — run_query  (custom — captures rows + SQL into state)
    # Errors are returned to generate_query as the tool result so it can fix
    # the SQL and retry.
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
    # NODE 4 — chart_config
    #
    # Receives actual sql_rows + sql_columns from state.
    # Asks the fast model (structured output) for the best chart type and columns.
    # Skips the LLM call when the query returned no rows. Otherwise always
    # produces a chart config — wants_chart=True is the only reason this node
    # runs, and even a single-value bar chart is valid.
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
"""

    class ChartConfig(BaseModel):
        chart_type: Literal["bar", "line", "pie", "scatter", "histogram"]
        x_col: str = Field(description="Exact column name for the x-axis / labels.")
        y_col: str = Field(description="Exact column name for the numeric values.")
        title: str = Field(description="Concise descriptive chart title.")
        x_label: str = Field(description="X-axis label.")
        y_label: str = Field(description="Y-axis label.")

    chart_configurator = fast_model.with_structured_output(ChartConfig)

    def chart_config(state: AgentState) -> dict:
        rows = state.get("sql_rows", [])
        columns = state.get("sql_columns", [])
        sql = state.get("last_sql", "")

        if not rows:
            logger.info("chart_config: no rows to plot, skipping")
            return {"chart_config": None}

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

Decide the best chart config."""

        try:
            cfg = chart_configurator.invoke(
                [
                    {"role": "system", "content": CHART_CONFIG_SYSTEM},
                    {"role": "user", "content": prompt},
                ]
            ).model_dump()
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
    ) -> Literal["run_query", "chart_config", "__end__"]:
        last = state["messages"][-1]
        # Still has a tool call → keep running SQL
        if getattr(last, "tool_calls", None):
            return "run_query"
        # Final answer — route based on wants_chart flag (set by analyze_question)
        if state.get("wants_chart", False):
            return "chart_config"
        return END

    # ══════════════════════════════════════════════════════════════════════════
    # Assemble graph
    # ══════════════════════════════════════════════════════════════════════════
    builder = StateGraph(AgentState)

    builder.add_node("analyze_question", analyze_question)
    builder.add_node("generate_query", generate_query)
    builder.add_node("run_query", run_query)
    builder.add_node("chart_config", chart_config)

    builder.add_edge(START, "analyze_question")
    builder.add_edge("analyze_question", "generate_query")
    builder.add_conditional_edges("generate_query", after_generate)
    builder.add_edge("run_query", "generate_query")
    builder.add_edge("chart_config", END)

    agent = builder.compile()
    logger.info(
        "Agent compiled: analyze→generate⇄[run]→[chart]→END"
    )
    return agent, db


# ══════════════════════════════════════════════════════════════════════════════
# run_agent
# ══════════════════════════════════════════════════════════════════════════════


async def run_agent(agent: Any, question: str, session_id: str) -> dict:
    """
    Run one agent turn. Returns answer, sql, chart_config, and metadata.
    """
    history = get_history_as_langchain_messages(
        session_id,
        limit=settings.history_messages,
        max_chars=settings.history_max_chars,
    )
    logger.info(f"[{session_id}] {len(history)} history msgs loaded.")

    initial: AgentState = {
        "messages": history + [HumanMessage(content=question)],
        "wants_chart": False,
        "last_sql": "",
        "sql_rows": [],
        "sql_columns": [],
        "chart_config": None,
    }

    try:
        final: AgentState = await agent.ainvoke(
            initial, config={"recursion_limit": RECURSION_LIMIT}
        )
    except GraphRecursionError:
        logger.warning(f"[{session_id}] gave up after {RECURSION_LIMIT} graph steps")
        final = {
            **initial,
            "messages": initial["messages"]
            + [AIMessage(content=GIVE_UP_ANSWER)],
            "wants_chart": False,
        }

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
