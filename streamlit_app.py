"""
streamlit_app.py - Conversational BI Agent UI

Run:
    streamlit run streamlit_app.py
"""

import base64
import os
import uuid
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
CHART_ENDPOINT = f"{API_BASE}/chart"
HISTORY_ENDPOINT = f"{API_BASE}/history"
HEALTH_ENDPOINT = f"{API_BASE}/health"
TABLES_ENDPOINT = f"{API_BASE}/tables"
REQUEST_TIMEOUT = 120

st.set_page_config(
    page_title="BI Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = ""
if "loaded_session" not in st.session_state:
    st.session_state.loaded_session = None


# ── Helpers ───────────────────────────────────────────────────────────────────


@st.cache_data(ttl=15, show_spinner=False)
def _check_health() -> dict | None:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_tables() -> list[str]:
    try:
        r = requests.get(TABLES_ENDPOINT, timeout=5)
        return r.json().get("tables", []) if r.status_code == 200 else []
    except Exception:
        return []


def _clear_history(session_id: str):
    try:
        requests.delete(f"{HISTORY_ENDPOINT}/{session_id}", timeout=10)
    except Exception:
        pass


def _fetch_history(session_id: str) -> list[dict]:
    try:
        r = requests.get(f"{HISTORY_ENDPOINT}/{session_id}", timeout=10)
        if r.status_code != 200:
            return []
        return [
            {"role": h["role"], "content": h["content"]}
            for h in r.json().get("history", [])
        ]
    except Exception:
        return []


def _new_session():
    # Runs before the rerun, so it may set the text_input's value
    st.session_state.session_id = f"session-{uuid.uuid4().hex[:6]}"


def _b64_to_image(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64)))


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 BI Agent")
    st.caption("Conversational Business Intelligence")
    st.divider()

    # Health
    health = _check_health()
    if health:
        st.success(f"● API connected — model: `{health.get('model','')}`")
    else:
        st.error("● API unreachable")
        st.caption("Run: `uvicorn app.main:app --reload`")

    st.divider()

    # Session ID — typed manually or generated with "New ID"
    session_id = st.text_input(
        "Session ID",
        key="session_id",
        help="Each session keeps its own conversation memory.",
    ).strip()

    col1, col2 = st.columns(2)
    with col1:
        st.button("🔄 New ID", use_container_width=True, on_click=_new_session)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True, disabled=not session_id):
            _clear_history(session_id)
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # Tables
    st.markdown("**Database Tables**")
    tables = _fetch_tables()
    if tables:
        for t in tables:
            st.markdown(f"• `{t}`")
    else:
        st.caption("Start API to see tables")

    st.divider()

    st.caption("LangGraph · FastAPI · Streamlit")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🧠 Conversational BI Agent")

if not session_id:
    st.error("Enter a Session ID in the sidebar before sending messages.")
    st.stop()

# Switching sessions shows that session's saved conversation
if st.session_state.loaded_session != session_id:
    st.session_state.messages = _fetch_history(session_id)
    st.session_state.loaded_session = session_id

# ── Replay chat history ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # SQL expander
            if msg.get("sql"):
                with st.expander("🔍 SQL Query", expanded=False):
                    st.code(msg["sql"], language="sql")
                    if msg.get("rewritten_question") and msg[
                        "rewritten_question"
                    ] != msg.get("original_q"):
                        st.caption(f"✏️ Rewritten as: *{msg['rewritten_question']}*")

            # Chart
            if msg.get("chart_b64"):
                try:
                    st.image(
                        _b64_to_image(msg["chart_b64"]),
                        caption=msg.get("chart_title", ""),
                        use_container_width=True,
                    )
                except Exception:
                    st.warning("Chart could not be rendered.")

            # Debug info
            with st.expander("🛠️ Debug Info", expanded=False):
                st.write("**Rewritten Question:**", msg.get("rewritten_question", "—"))
                st.write("**Wants Chart:**", msg.get("wants_chart", False))
                st.write("**Chart Type:**", msg.get("chart_type") or "—")
                st.write("**Chart Title:**", msg.get("chart_title") or "—")
                st.write("**SQL Rows Returned:**", msg.get("sql_rows_count", "—"))


# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask anything about your data…"):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call /chart endpoint
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response = requests.post(
                    CHART_ENDPOINT,
                    json={"message": prompt, "session_id": session_id},
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Is the FastAPI server running?")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try a simpler question.")
                st.stop()

        if response.status_code == 200:
            data = response.json()

            answer = data.get("answer") or "No response."
            sql = data.get("sql", "")
            rewritten_q = data.get("rewritten_question", "")
            wants_chart = data.get("wants_chart", False)
            chart = data.get("chart") or {}
            chart_b64 = chart.get("image_base64") if chart else None
            chart_type = chart.get("chart_type", "") if chart else ""
            chart_title = chart.get("title", "") if chart else ""
            chart_rows = chart.get("rows", 0) if chart else 0

            # Answer text
            st.markdown(answer)

            # SQL
            if sql:
                with st.expander("🔍 SQL Query", expanded=False):
                    st.code(sql, language="sql")
                    if rewritten_q and rewritten_q != prompt:
                        st.caption(f"✏️ Rewritten as: *{rewritten_q}*")

            # Chart
            if chart_b64:
                try:
                    st.image(
                        _b64_to_image(chart_b64),
                        caption=chart_title,
                        use_container_width=True,
                    )
                except Exception:
                    st.warning("Chart image could not be rendered.")
            elif wants_chart:
                st.info(
                    "ℹ️ A chart was requested but could not be generated for this data."
                )

            # Debug
            with st.expander("🛠️ Debug Info", expanded=False):
                st.write("**Rewritten Question:**", rewritten_q or "—")
                st.write("**Wants Chart:**", wants_chart)
                st.write("**Chart Type:**", chart_type or "—")
                st.write("**Chart Title:**", chart_title or "—")
                st.write("**SQL Rows Returned:**", chart_rows or "—")

            # Save to history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sql": sql,
                    "rewritten_question": rewritten_q,
                    "original_q": prompt,
                    "wants_chart": wants_chart,
                    "chart_b64": chart_b64,
                    "chart_type": chart_type,
                    "chart_title": chart_title,
                    "sql_rows_count": chart_rows,
                }
            )

        else:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            st.error(f"Agent request failed ({response.status_code}): {detail}")
