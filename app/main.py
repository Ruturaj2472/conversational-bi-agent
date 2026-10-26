"""
app/main.py - FastAPI application for the Conversational BI Agent v4
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.memory import init_db, close_db, get_history, clear_history
from app.agent import build_agent, run_agent
from charts.generator import render_chart
from mlflow_track.tracker import init_mlflow, track_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

_agent = None
_db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent, _db
    logger.info("Starting Conversational BI Agent...")
    init_mlflow()
    init_db()
    _agent, _db = build_agent()
    logger.info("Agent ready.")
    yield
    close_db()
    logger.info("Agent shut down.")


app = FastAPI(
    title="Conversational BI Agent",
    description="Chat with your database in plain English.",
    version="4.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default")


class ChartMeta(BaseModel):
    chart_type: str
    title: str
    x_col: str
    y_col: str
    rows: int
    image_base64: str


class ChatResponse(BaseModel):
    answer: str
    sql: str
    session_id: str
    rewritten_question: str
    wants_chart: bool


class ChartResponse(BaseModel):
    answer: str
    sql: str
    session_id: str
    rewritten_question: str
    wants_chart: bool
    chart: Optional[ChartMeta] = None


class HistoryEntry(BaseModel):
    id: int
    role: str
    content: str
    created_at: Optional[str] = None


class HistoryResponse(BaseModel):
    session_id: str
    count: int
    history: list[HistoryEntry]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok", "model": settings.model_name, "version": "4.0.0"}


@app.get("/tables", tags=["System"])
async def list_tables():
    if _db is None:
        raise HTTPException(503, "Agent not ready.")
    return {"tables": _db.get_usable_table_names()}


# Agent, chart and memory endpoints are plain `def`: FastAPI runs them in its
# thread pool, so blocking LLM / DB / matplotlib work never stalls the event loop.


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
def chat(req: ChatRequest):
    """Text answer + SQL. No chart."""
    if _agent is None:
        raise HTTPException(503, "Agent not ready.")
    with track_query(session_id=req.session_id, question=req.message) as tracker:
        try:
            result = run_agent(_agent, req.message, req.session_id)
            tracker.log_result(result)
        except Exception as exc:
            logger.error(f"Agent error: {exc}", exc_info=True)
            raise HTTPException(500, f"Agent error: {exc}")
    return ChatResponse(
        answer=result["answer"],
        sql=result["sql"],
        session_id=req.session_id,
        rewritten_question=result["rewritten_q"],
        wants_chart=result["wants_chart"],
    )


@app.post("/chart", response_model=ChartResponse, tags=["Agent"])
def chat_with_chart(req: ChatRequest):
    """
    Text answer + SQL + chart image.
    - intent_detector decides if user wants a chart (explicit or implied).
    - chart_config node picks chart type + axes from actual SQL result rows.
    - render_chart renders the PNG.
    - chart=null if not applicable or render failed.
    """
    if _agent is None:
        raise HTTPException(503, "Agent not ready.")

    with track_query(session_id=req.session_id, question=req.message) as tracker:
        try:
            result = run_agent(_agent, req.message, req.session_id)
            tracker.log_result(result)
        except Exception as exc:
            logger.error(f"Agent error: {exc}", exc_info=True)
            raise HTTPException(500, f"Agent error: {exc}")

        chart_out = None
        cfg = result.get("chart_config")
        if cfg:
            try:
                rendered = render_chart(
                    rows=result["sql_rows"],
                    columns=result["sql_columns"],
                    config=cfg,
                )
                if rendered.get("image_base64"):
                    chart_out = ChartMeta(
                        chart_type=rendered["chart_type"],
                        title=rendered["title"],
                        x_col=rendered["x_col"],
                        y_col=rendered["y_col"],
                        rows=rendered["rows"],
                        image_base64=rendered["image_base64"],
                    )
                    tracker.log_chart(rendered)
                else:
                    logger.warning(
                        f"render_chart returned no image: {rendered.get('error')}"
                    )
            except Exception as exc:
                logger.warning(f"Chart render failed (non-critical): {exc}")

    return ChartResponse(
        answer=result["answer"],
        sql=result["sql"],
        session_id=req.session_id,
        rewritten_question=result["rewritten_q"],
        wants_chart=result["wants_chart"],
        chart=chart_out,
    )


@app.get("/history/{session_id}", response_model=HistoryResponse, tags=["Memory"])
def get_conversation_history(session_id: str, limit: int = 50):
    rows = get_history(session_id, limit=limit)
    return HistoryResponse(
        session_id=session_id,
        count=len(rows),
        history=[HistoryEntry(**r) for r in rows],
    )


@app.delete("/history/{session_id}", tags=["Memory"])
def delete_conversation_history(session_id: str):
    deleted = clear_history(session_id)
    return {
        "message": f"Cleared {deleted} messages for session '{session_id}'.",
        "deleted": deleted,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
