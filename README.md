# 🧠 Conversational Business Intelligence Agent

A full-stack BI agent that lets you query your database in plain English, get clean answers, and generate charts — all from a chat interface.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [Agent Graph](#5-agent-graph)
6. [Performance](#6-performance)
7. [Prerequisites](#7-prerequisites)
8. [Running with Docker](#8-running-with-docker)
9. [Running Locally (without Docker)](#9-running-locally-without-docker)
10. [Environment Variables](#10-environment-variables)
11. [API Reference](#11-api-reference)
12. [Testing](#12-testing)
13. [Evaluation](#13-evaluation)
14. [Continuous Integration](#14-continuous-integration)
15. [Example Questions](#15-example-questions)

---

## 1. Project Overview

This system lets business users ask natural language questions about their data and receive:

- A clean, plain-English **answer**
- The exact **SQL query** that was executed
- An AI-generated **chart** (bar, line, pie, scatter, histogram) when relevant
- Full **conversation memory** — follow-up questions resolve context automatically

### Key capabilities

| Capability | Details |
|---|---|
| Natural language to SQL | LangGraph agent writes and executes SQL, and fixes its own query when the database returns an error |
| Read-only execution | Agent SQL runs in a read-only transaction with a statement timeout |
| Conversational memory | PostgreSQL-backed, persists across sessions |
| Query rewriting + chart intent | One structured-output call resolves follow-ups ("that category" → "Beverages") and decides whether a chart helps |
| Clarifying questions | Asks the user to clarify instead of guessing when a question refers to context that doesn't exist |
| AI chart configuration | LLM picks chart type and axes from the actual result rows (structured output) |
| MLflow tracing | Every question is logged as a run; every LLM call is traced with tokens and cost |

---

## 2. Architecture

```
                    ┌─────────────────────────────────┐
                    │         Streamlit UI             │
                    │      (http://localhost:8501)     │
                    └──────────────┬──────────────────┘
                                   │ HTTP POST /chart
                    ┌──────────────▼──────────────────┐
                    │         FastAPI Backend          │
                    │      (http://localhost:8000)     │
                    │                                  │
                    │   ┌──────────────────────────┐   │
                    │   │    LangGraph SQL Agent   │   │
                    │   │                          │   │
                    │   │  analyze_question        │   │
                    │   │  generate_query          │   │
                    │   │  run_query               │   │
                    │   │  chart_config            │   │
                    │   └──────────────────────────┘   │
                    └──────┬───────────────┬───────────┘
                           │               │
           ┌───────────────▼───┐     ┌─────▼──────────────┐
           │  PostgreSQL        │     │  PostgreSQL         │
           │  Business DB       │     │  Memory DB          │
           │  (northwind)       │     │  (agent_memory)     │
           └───────────────────┘     └────────────────────┘

                    ┌─────────────────────────┐
                    │   MLflow Tracking        │
                    │  (http://localhost:5000) │
                    └─────────────────────────┘
```

With Docker Compose, all of these — PostgreSQL, MLflow, the FastAPI backend and the Streamlit UI — run as containers.

---

## 3. Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | OpenAI GPT-4o (SQL + answers) and GPT-4.1-mini (question analysis + chart config), both configurable |
| Business Database | PostgreSQL 16 (Northwind) |
| Conversation Memory | PostgreSQL via SQLAlchemy (custom, no LangGraph checkpointer) |
| API Layer | FastAPI |
| Chart Generation | Matplotlib (thread-safe `Figure` API) + Pandas |
| Experiment Tracking | MLflow |
| Frontend | Streamlit |
| Containerisation | Docker + Docker Compose |
| Tests, Evaluation & CI | pytest, ruff, GitHub Actions |

---

## 4. Project Structure

```
conversational-bi-agent/
│
├── streamlit_app.py            # Streamlit chat UI
│
├── app/
│   ├── main.py                 # FastAPI app — endpoints /chat /chart /history /tables /health
│   ├── agent.py                # LangGraph agent — 4 nodes + run_agent()
│   ├── sql_runner.py           # Read-only SQL execution, returns real columns + rows
│   ├── memory.py               # SQLAlchemy conversation store
│   └── config.py               # Pydantic settings — reads from .env
│
├── charts/
│   └── generator.py            # Pure chart renderer — takes rows + config → PNG
│
├── mlflow_track/
│   └── tracker.py              # Non-blocking MLflow run tracker
│
├── scripts/
│   ├── load_northwind.py       # Loads Northwind into Postgres (used by Docker Compose)
│   └── load_chinook.py         # Alternative: Chinook music store DB
│
├── tests/                      # pytest suite (no OpenAI key or Postgres needed)
│   └── integration/            # Tests against a real PostgreSQL (skipped without one)
│
├── evals/
│   ├── questions.yaml          # Evaluation set — questions + reference SQL + expectations
│   ├── scoring.py              # Execution accuracy, chart and clarification checks
│   ├── run_eval.py             # Scores a recorded run (--replay) or a running API (--live)
│   ├── baseline.json           # Recorded scores + thresholds enforced in CI
│   └── results/
│       └── optimized_run.json  # Recorded run: answers, SQL, latency, LLM calls, tokens, cost
│
├── .github/workflows/ci.yml    # CI: lint, tests, Docker builds, secret scan, evaluation
├── pyproject.toml              # ruff + pytest settings
│
├── docker/postgres/
│   └── 01-create-memory-db.sh  # Creates the agent_memory DB on first Postgres start
│
├── Dockerfile                  # FastAPI container (also used for MLflow and the data loader)
├── Dockerfile.streamlit        # Streamlit container
├── docker-compose.yml          # Full stack: postgres, db-loader, mlflow, bi-agent, bi-ui
├── requirements.txt            # API dependencies (pinned)
├── requirements-ui.txt         # Streamlit UI dependencies (pinned)
├── requirements-dev.txt        # Everything + pytest, ruff
└── .env.example                # Environment variable template
```

---

## 5. Agent Graph

Every user message flows through this LangGraph pipeline:

```
START
  │
  ▼
[analyze_question]    One structured-output call on the fast model (gpt-4.1-mini) that:
  │                   - rewrites follow-ups into self-contained questions using history,
  │                     e.g. "how many products are in that category?"
  │                     → "How many products are in the Beverages category?"
  │                   - decides wants_chart, from explicit ("show me a bar chart") or
  │                     implied intent ("compare sales by region")
  ▼
[generate_query]      Writes SQL + the natural language answer (gpt-4o).
  │                   The full database schema (tables, columns, sample rows) is loaded
  │                   once at startup and placed at the start of the system prompt, so
  │                   OpenAI's prompt cache reuses it across requests.
  │                   Sees only the current (rewritten) question, not the whole history.
  │                   In CHART MODE: writes a GROUP BY query returning ≥2 columns.
  │
  ├── (tool call) ──► [run_query] ──► back to [generate_query]
  │                    Executes the SQL in a read-only transaction and stores the real
  │                    column names + rows in state. Database errors are returned to
  │                    generate_query, which fixes the query and retries
  │                    (capped by the graph's recursion limit).
  │
  └── (final answer)
          │
          ├── wants_chart=True  ──► [chart_config] ──► END
          │                         gpt-4.1-mini decides chart_type, x_col, y_col,
          │                         title and labels from the actual rows
          │                         (skipped when the query returned no rows)
          │
          └── wants_chart=False ──► END
```

LLM calls per question: **2** (plain question) to **4** (chart question), plus 1 per SQL retry.

**Memory flow (outside the graph):**
- Before: the last `HISTORY_MESSAGES` messages are loaded from the `agent_memory` DB (long answers truncated to `HISTORY_MAX_CHARS`)
- After: the user question and final answer are saved to `agent_memory` in one transaction

---

## 6. Performance

Measured end to end on the Docker stack (PostgreSQL 16 + Northwind) with the 11 questions in `example_queries.txt`, comparing the original version of this project (commit `4c85659`) with the current one. Both used the same database, the same library versions and the same models (gpt-4o + gpt-4.1-mini). LLM calls, tokens and cost come from MLflow traces; answers were checked against the database.

| Metric | Original | Current |
|---|---|---|
| Average response time | 8.1 s | **5.4 s** (−33%) |
| Median / p95 response time | 7.9 s / 10.6 s | **5.5 s / 7.3 s** |
| LLM calls (11 questions) | 62 | **37** (−40%) |
| gpt-4o calls | 59 | **20** (−66%) |
| LLM calls per question | 5–7 | **2–4** |
| Output tokens | 2,800 | **1,954** (−30%) |
| Input tokens served from prompt cache | 14% | **85%** |
| Correct and complete answers | 7 / 11 | **11 / 11** |
| Charts produced for chart questions | 2 / 7 | **6 / 7** |
| OpenAI cost per question | $0.0094 | $0.0105 |

**Cost trade-off:** the current version sends the full schema (14 tables, ~2,700 tokens) with every gpt-4o call instead of looking up only the relevant tables with an extra LLM call. That removes round trips — which is where the latency gain comes from — but adds input tokens. Most of them are served from OpenAI's prompt cache at a discount, and the result is about $0.001 more per question. For much larger schemas, selecting relevant tables first (e.g. by keyword or embedding search) would keep the prompt small without adding an LLM call.

These figures come from one run of 11 questions per version; LLM latency varies between runs, so treat percentages as approximate.

---

## 7. Prerequisites

- Docker Desktop (includes Docker Compose) — for the Docker option
- Python 3.11+ and PostgreSQL 14+ — only for running without Docker
- OpenAI API key

---

## 8. Running with Docker

Docker Compose runs the whole stack: PostgreSQL, a one-off Northwind loader, MLflow, the FastAPI backend and the Streamlit UI.

### Step 1 — Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```env
OPENAI_API_KEY=sk-your-key-here
POSTGRES_PASSWORD=choose-a-password
```

Hosts are set by `docker-compose.yml` — you don't need to change `POSTGRES_HOST` or `MLFLOW_TRACKING_URI`.

### Step 2 — Build and start

```bash
docker compose up --build
```

On first start:
1. `postgres` creates the `northwind` and `agent_memory` databases
2. `db-loader` downloads Northwind and loads it, then exits
3. `mlflow` and `bi-agent` start; `bi-ui` starts once the API is healthy

The first build takes a few minutes. Later starts are fast, and data persists in Docker volumes.

### Step 3 — Access services

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| PostgreSQL | `localhost:5432` (user/password from `.env`) |

### Docker management commands

```bash
# Start all services (background)
docker compose up -d

# Container status
docker compose ps

# View live logs
docker compose logs -f
docker compose logs -f bi-agent

# Rebuild and restart one service after a code change
docker compose up -d --build bi-agent

# Query the business database
docker exec -it bi-postgres psql -U postgres -d northwind

# Stop all
docker compose down

# Stop and delete all data (Postgres + MLflow volumes)
docker compose down -v
```

### How containers communicate

```
bi-ui      ──► bi-agent:8000      (FastAPI)
bi-agent   ──► postgres:5432      (northwind + agent_memory)
bi-agent   ──► mlflow:5000        (tracking)
db-loader  ──► postgres:5432      (one-off data load)
```

---

## 9. Running Locally (without Docker)

### Step 1 — Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-ui.txt
```

### Step 2 — Configure environment

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-your-key-here
POSTGRES_HOST=localhost
POSTGRES_PASSWORD=yourpassword
POSTGRES_DB=northwind
```

`MEMORY_POSTGRES_*` values that are not set fall back to the `POSTGRES_*` values.

### Step 3 — Create the databases and load Northwind

```bash
psql -U postgres -c "CREATE DATABASE agent_memory;"
python scripts/load_northwind.py     # creates 'northwind' if needed and loads it
```

Expected output (end):
```
INFO | Tables created:
INFO |   categories                          8 rows
INFO |   customers                          91 rows
INFO |   order_details                    2155 rows
INFO |   orders                            830 rows
INFO |   products                           77 rows
INFO |   suppliers                          29 rows
INFO |   ...
INFO | ✅ Northwind loaded successfully!
```

The table `conversation_messages` in `agent_memory` is created automatically when the API starts.

### Step 4 — Start MLflow (optional)

```bash
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

If MLflow is not running, tracking is skipped and the app works normally.

### Step 5 — Start the FastAPI backend

```bash
uvicorn app.main:app --reload --port 8000
```

You should see:
```
INFO | DB: dialect=postgresql tables=['categories', 'customers', ...]
INFO | Schema loaded: 14 tables, ... chars
INFO | Agent compiled: analyze→generate⇄[run]→[chart]→END
INFO | Agent ready.
```

### Step 6 — Start the Streamlit UI

Open a second terminal:
```bash
streamlit run streamlit_app.py
# Opens http://localhost:8501
```

### About the databases

**Northwind** is the business database — all tables are in the `public` schema:

| Table | Description |
|---|---|
| `products` | Product catalogue, prices, stock levels, reorder points |
| `categories` | Product categories (Beverages, Condiments, Seafood, etc.) |
| `suppliers` | Vendor information, country, contact |
| `orders` | Sales orders with dates, freight, ship info |
| `order_details` | Line items — product, quantity, unit price, discount |
| `customers` | Customer info, city, country |
| `employees` | Staff, territory, hire date |
| `shippers` | Shipping companies |

**agent_memory** is managed entirely by the app.

---

## 10. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key — the app fails at startup if it is missing |
| `POSTGRES_HOST` | `localhost` | Business DB host (set to `postgres` by Docker Compose) |
| `POSTGRES_PORT` | `5432` | Business DB port |
| `POSTGRES_USER` | `postgres` | Business DB user |
| `POSTGRES_PASSWORD` | `postgres` | Business DB password |
| `POSTGRES_DB` | `northwind` | Business DB name |
| `MEMORY_POSTGRES_HOST` | `POSTGRES_HOST` | Memory DB host |
| `MEMORY_POSTGRES_PORT` | `POSTGRES_PORT` | Memory DB port |
| `MEMORY_POSTGRES_USER` | `POSTGRES_USER` | Memory DB user |
| `MEMORY_POSTGRES_PASSWORD` | `POSTGRES_PASSWORD` | Memory DB password |
| `MEMORY_POSTGRES_DB` | `agent_memory` | Memory DB name |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL (use `http://localhost:5000` locally) |
| `MLFLOW_EXPERIMENT_NAME` | `bi-agent` | MLflow experiment name |
| `MODEL_NAME` | `gpt-4.1-mini` | Model for SQL generation and answers (`.env.example` uses `gpt-4o`) |
| `FAST_MODEL_NAME` | `gpt-4.1-mini` | Model for question analysis and chart config |
| `MAX_RESULTS` | `10` | Default N for top-N / ranking questions |
| `QUERY_TIMEOUT_SECONDS` | `30` | PostgreSQL statement timeout for agent queries |
| `HISTORY_MESSAGES` | `10` | Past messages used to resolve follow-up questions |
| `HISTORY_MAX_CHARS` | `500` | Longer past answers are truncated in the prompt |
| `API_BASE_URL` | `http://localhost:8000` | Streamlit UI → API address (set by Docker Compose) |
| `DEBUG` | `false` | Enable SQLAlchemy SQL echo |

---

## 11. API Reference

### `POST /chart`
Main endpoint — returns answer + SQL + optional chart.

**Request:**
```json
{
  "message": "Show top 5 products by revenue as a bar chart",
  "session_id": "user-abc123"
}
```

**Response:**
```json
{
  "answer": "The top 5 products by revenue are Côte de Blaye ($141,396), ...",
  "sql": "SELECT p.product_name, SUM(od.unit_price * od.quantity * (1 - od.discount)) AS revenue ...",
  "session_id": "user-abc123",
  "rewritten_question": "Show the top 5 products by revenue as a bar chart",
  "wants_chart": true,
  "chart": {
    "chart_type": "bar",
    "title": "Top 5 Products by Revenue",
    "x_col": "product_name",
    "y_col": "revenue",
    "rows": 5,
    "image_base64": "iVBORw0KGgoAAAANS..."
  }
}
```

### `POST /chat`
Text answer + SQL only — no chart generation.

### `GET /history/{session_id}`
Returns conversation history for a session (`?limit=50` by default).

### `DELETE /history/{session_id}`
Clears conversation history for a session.

### `GET /tables`
Lists all available tables in the connected database.

### `GET /health`
Health check — returns model name and version.

---

## 12. Testing

The test suite runs without an OpenAI key or PostgreSQL: it uses SQLite and a scripted fake LLM to run the full LangGraph agent.

```bash
pip install -r requirements-dev.txt
pytest
```

| File | Covers |
|---|---|
| `tests/test_config.py` | Defaults, memory-DB fallback, URL escaping, required API key |
| `tests/test_sql_runner.py` | Real column names, row cap, errors, rollback, type conversion |
| `tests/test_memory.py` | Ordered batch writes, latest-N history, truncation, clearing |
| `tests/test_charts.py` | Every chart type, non-numeric values, date labels, concurrent rendering |
| `tests/test_agent_graph.py` | End-to-end graph runs: chart flow, schema in prompt, SQL retry, retry cap, empty results |
| `tests/test_eval_scoring.py` | Evaluation scoring rules and consistency of the `evals/` files |

### Integration tests (real PostgreSQL)

`tests/integration/` runs against PostgreSQL with Northwind loaded and is skipped unless the database URLs are set:

```bash
# with the Docker stack running
export INTEGRATION_DB_URL=postgresql+psycopg2://postgres:<password>@localhost:5432/northwind
export INTEGRATION_MEMORY_DB_URL=postgresql+psycopg2://postgres:<password>@localhost:5432/agent_memory
pytest tests/integration
```

They check that writes (`DELETE`, `DROP`) are rejected by the database, that the statement timeout stops slow queries, numeric/date conversion, the row cap, every reference query in `evals/questions.yaml`, and the memory store with its index.

---

## 13. Evaluation

`evals/` measures answer quality with **execution accuracy**: the agent's SQL and a hand-written reference query are both executed on PostgreSQL, and the results must match (column names and extra columns are ignored; row order is checked where the question implies one). No LLM is used for scoring.

Each question in `evals/questions.yaml` has a category and expectations:

| Check | Passes when |
|---|---|
| Execution | The agent's result matches the reference result (`set`, `ordered`, or `top_row` match) |
| Clarification | For ambiguous questions ("Who supplies *those* products?" with no context), the agent asks instead of running SQL |
| Chart | A chart of the expected type is produced when one is expected, and none when it isn't |

### Current scores

Recorded run of the 11 questions (`evals/results/optimized_run.json`), gpt-4o + gpt-4.1-mini:

| Metric | Score |
|---|---|
| Questions passed | **11 / 11** |
| Execution accuracy | **100%** (9 / 9) |
| Clarification accuracy | **100%** (2 / 2) |
| Chart accuracy | **100%** (7 / 7) |
| Errors | 0 |
| Avg / p95 latency | 5.4 s / 7.4 s |
| Avg LLM calls per question | 3.4 |
| Avg cost per question | $0.0105 |

### Running the evaluation

```bash
# Score the recorded run (no OpenAI calls; needs PostgreSQL with Northwind)
python -m evals.run_eval

# Fail with exit code 1 if a threshold in evals/baseline.json is not met
python -m evals.run_eval --check

# Ask a running API every question, score the answers and save them as a new run
# (calls OpenAI through the agent)
python -m evals.run_eval --live --api http://localhost:8000 --save evals/results/live_run.json
```

Database connection comes from `EVAL_DB_URL` or the `POSTGRES_*` variables.

Thresholds (`evals/baseline.json`): execution accuracy ≥ 85%, clarification accuracy 100%, chart accuracy ≥ 85%, 0 errors, ≤ 4 LLM calls and ≤ $0.012 per question on average.

---

## 14. Continuous Integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request. None of the jobs call OpenAI.

| Job | What it checks |
|---|---|
| `lint-and-unit-tests` | `ruff check` and the unit test suite |
| `docker` | Both images build; `docker-compose.yml` is valid |
| `secret-scan` | gitleaks scans the git history for committed secrets |
| `integration-and-eval` | PostgreSQL 16 service + Northwind: integration tests, then `python -m evals.run_eval --check` on the recorded run; the report is uploaded as an artifact |

---

## 15. Example Questions

From `example_queries.txt` (questions in the same group share a session, so follow-ups work):

```
Which category had the highest revenue?
How many products are in that category?

Which country has the most suppliers? Show as a pie chart
Who are the top 5 employees by number of orders handled?

Which products are below their reorder level? List them with current stock and reorder point

Show me monthly order volume trend for 1997 as a line chart

Which supplier provides the most expensive products on average?

What are the top 3 most ordered products?
```
