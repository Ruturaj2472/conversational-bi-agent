# 🧠 Conversational Business Intelligence Agent

A full-stack BI agent that lets you query your database in plain English, get clean answers, and generate charts — all from a chat interface.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [Agent Graph](#5-agent-graph)
6. [Prerequisites](#6-prerequisites)
7. [Database Setup & Data Load](#7-database-setup--data-load)
8. [Running Locally](#8-running-locally)
9. [Running with Docker](#9-running-with-docker)
10. [Environment Variables](#10-environment-variables)
11. [API Reference](#11-api-reference)
12. [Example Questions](#12-example-questions)

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
| Natural language to SQL | LangGraph agent writes, checks, and executes SQL |
| Conversational memory | PostgreSQL-backed, persists across sessions |
| Query rewriting | Resolves pronouns and follow-ups using history |
| Chart intent detection | Detects when user wants a chart, even implicitly |
| AI chart configuration | LLM picks chart type and axes from actual result data |
| MLflow tracing | Every node, LLM call, and SQL execution is traced |

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
                    │   │  query_rewriter          │   │
                    │   │  intent_detector         │   │
                    │   │  list_tables             │   │
                    │   │  call_get_schema         │   │
                    │   │  generate_query          │   │
                    │   │  check_query             │   │
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

---

## 3. Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM | OpenAI GPT-4.1-mini |
| Business Database | PostgreSQL (Northwind) |
| Conversation Memory | PostgreSQL via SQLAlchemy (custom, no LangGraph checkpointer) |
| API Layer | FastAPI |
| Chart Generation | Matplotlib + Pandas |
| Experiment Tracking | MLflow |
| Frontend | Streamlit |
| Containerisation | Docker + Docker Compose |

---

## 4. Project Structure

```
bi-agent/
│
├── streamlit_app.py          # Streamlit chat UI
│
├── app/
│   ├── main.py               # FastAPI app — endpoints /chat /chart /history
│   ├── agent.py              # LangGraph agent — all 8 nodes + run_agent()
│   ├── memory.py             # SQLAlchemy conversation store
│   └── config.py             # Pydantic settings — reads from .env
│
├── charts/
│   └── generator.py          # Pure chart renderer — takes rows + config → PNG
│
├── mlflow/
│   └── tracker.py            # Async-safe MLflow RunTracker
│
├── scripts/
│   ├── load_northwind.py     # ✅ Recommended: loads Northwind into Postgres
│   ├── load_chinook.py       # Alternative: Chinook music store DB
│   ├── load_adventureworks.py # Alternative: Adventure Works (complex)
│   └── test_agent.py         # CLI smoke test for the agent
│
├── Dockerfile                # FastAPI container
├── Dockerfile.streamlit      # Streamlit container
├── docker-compose.yml        # All services (agent + streamlit + mlflow)
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variable template
```

---

## 5. Agent Graph

Every user message flows through this LangGraph pipeline:

```
START
  │
  ▼
[query_rewriter]      Rewrites ambiguous follow-ups into self-contained questions
  │                   using conversation history. E.g. "what about that one?"
  │                   → "What is the total revenue for the Beverages category?"
  ▼
[intent_detector]     Classifies whether the user wants a chart (YES/NO).
  │                   Detects explicit ("show me a bar chart") and implied intent
  │                   ("compare sales by region" → chart useful → YES).
  ▼
[list_tables]         Forced tool call — lists all available DB tables.
  │
  ▼
[call_get_schema]     Forces schema retrieval for relevant tables.
  │
  ▼
[get_schema]          ToolNode — executes the schema call.
  │
  ▼
[generate_query]      Writes SQL + natural language answer.
  │                   In CHART MODE: writes a GROUP BY query returning ≥2 columns.
  │                   Appends CHART_REQUEST token if chart is needed.
  │
  ├── (tool call) ──► [check_query] ──► [run_query] ──► back to [generate_query]
  │                    SQL expert         Executes SQL,
  │                    double-checks      captures rows +
  │                    the query          columns into state
  │
  └── (final answer)
          │
          ├── wants_chart=True  ──► [chart_config] ──► END
          │                         gpt-4.1-mini decides:
          │                         chart_type, x_col, y_col,
          │                         title, labels — from actual rows
          │
          └── wants_chart=False ──► END
```

**Memory flow (outside the graph):**
- Before: history loaded from `agent_memory` DB → prepended to messages
- After: user question + final answer saved to `agent_memory` DB

---

## 6. Prerequisites

- Python 3.11+
- PostgreSQL 14+ running locally
- Docker + Docker Compose (for Docker option)
- OpenAI API key

---

## 7. Database Setup & Data Load

Two PostgreSQL databases are needed:

| Database | Purpose |
|---|---|
| `northwind` | Business data (products, orders, suppliers, customers) |
| `agent_memory` | Conversation history (auto-created by the app on startup) |

### Step 1 — Create the databases

```bash
psql -U postgres -c "CREATE DATABASE northwind;"
psql -U postgres -c "CREATE DATABASE agent_memory;"
```

### Step 2 — Load Northwind data

The loader script downloads the Northwind SQL file from GitHub and loads it into PostgreSQL automatically.

```bash
# Install loader dependencies if not already installed
pip install requests psycopg2-binary python-dotenv

# Run the loader
python scripts/load_northwind.py
```

Expected output:
```
INFO | Database 'northwind' already exists.
INFO | Downloading Northwind SQL from GitHub...
INFO |   ✓ Downloaded (84 KB)
INFO | Loading Northwind into 'northwind'...
INFO |   ✓ All statements executed.

INFO | Tables created:
INFO |   categories                        8 rows
INFO |   customers                        91 rows
INFO |   employee_territories             49 rows
INFO |   employees                         9 rows
INFO |   order_details                  2155 rows
INFO |   orders                           830 rows
INFO |   products                          77 rows
INFO |   regions                            4 rows
INFO |   shippers                           3 rows
INFO |   suppliers                         29 rows
INFO |   territories                       53 rows

INFO | ✅ Northwind loaded successfully!
```

### Step 3 — Verify

```bash
psql -U postgres -d northwind -c "\dt"
psql -U postgres -d northwind -c "SELECT COUNT(*) FROM orders;"
```

### About the databases

**Northwind** is the business database — all tables land in the `public` schema:

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

**agent_memory** is managed entirely by the app. The table `conversation_messages` is created automatically on first startup.

---

## 8. Running Locally

### Step 1 — Clone and install

```bash
cd to the zip folder

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install streamlit pillow     # extra deps for the UI
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
MEMORY_POSTGRES_PASSWORD=yourpassword
```

### Step 3 — Start MLflow

```bash
set MLFLOW_SERVER_ALLOWED_HOSTS=*
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow/artifacts
# Open http://localhost:5000
```

### Step 4 — Start the FastAPI backend

```bash
uvicorn app.main:app --reload --port 8000
```

You should see:
```
INFO | DB: dialect=postgresql tables=['categories', 'customers', ...]
INFO | Agent compiled: rewriter→intent→tables→schema→generate...
INFO | ✅ Agent ready.
```

### Step 5 — Start the Streamlit UI

Open a second terminal:
```bash
streamlit run streamlit_app.py
# Opens http://localhost:8501
```

### Local service URLs

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

---

## 9. Running with Docker

PostgreSQL stays on your **local machine** (outside Docker). The three containers — FastAPI, Streamlit, MLflow — run inside Docker and connect to your local Postgres via `host.docker.internal`.

### Step 1 — Make sure databases are loaded

Follow [Section 7](#7-database-setup--data-load) first. The databases must exist on your local Postgres before starting Docker.

### Step 2 — Configure `.env`

```bash
cp .env.example .env
```

Edit `.env` — use `host.docker.internal` as the Postgres host so containers can reach your local machine:

```env
OPENAI_API_KEY=sk-your-key-here
POSTGRES_HOST=host.docker.internal
POSTGRES_PASSWORD=yourpassword
POSTGRES_DB=northwind
MEMORY_POSTGRES_HOST=host.docker.internal
MEMORY_POSTGRES_PASSWORD=yourpassword
MLFLOW_TRACKING_URI=http://mlflow:5000
```

### Step 3 — Linux only: allow Docker to connect to local Postgres

On **Mac and Windows**, Docker Desktop handles `host.docker.internal` automatically — skip this step.

On **Linux**, you need to:

**a) Allow external connections in `postgresql.conf`:**
```bash
# Find your config file
psql -U postgres -c "SHOW config_file;"

# Add or update this line
listen_addresses = '*'
```

**b) Allow the Docker subnet in `pg_hba.conf`:**
```
host    all    all    172.16.0.0/12    md5
```

**c) Restart PostgreSQL:**
```bash
sudo systemctl restart postgresql
```

```bash
set MLFLOW_SERVER_ALLOWED_HOSTS=*
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow/artifacts
# Open http://localhost:5000
```

### Step 4 — Build and start

```bash
docker-compose up --build
```

First build takes 2–3 minutes. Subsequent starts are fast.

### Step 5 — Access services

| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

### Docker management commands

```bash
# Start all services (background)
docker-compose up -d

# Start with rebuild (after code changes)
docker-compose up --build

# View live logs
docker-compose logs -f

# View logs for one service
docker-compose logs -f bi-agent
docker-compose logs -f streamlit
docker-compose logs -f mlflow

# Restart one service after code change
docker-compose up --build bi-agent

# Stop all
docker-compose down

# Stop and wipe MLflow data
docker-compose down -v
```

### How containers communicate

```
bi-agent   ──► host.docker.internal:5432   (your local Postgres)
bi-agent   ──► mlflow:5000        (internal Docker network)
```

---

## 10. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `POSTGRES_HOST` | `localhost` | Business DB host. Use `host.docker.internal` in Docker |
| `POSTGRES_PORT` | `5432` | Business DB port |
| `POSTGRES_USER` | `postgres` | Business DB user |
| `POSTGRES_PASSWORD` | `postgres` | Business DB password |
| `POSTGRES_DB` | `northwind` | Business DB name |
| `MEMORY_POSTGRES_HOST` | `localhost` | Memory DB host |
| `MEMORY_POSTGRES_PORT` | `5432` | Memory DB port |
| `MEMORY_POSTGRES_USER` | `postgres` | Memory DB user |
| `MEMORY_POSTGRES_PASSWORD` | `postgres` | Memory DB password |
| `MEMORY_POSTGRES_DB` | `agent_memory` | Memory DB name |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | `bi-agent` | MLflow experiment name |
| `MODEL_NAME` | `gpt-4.1-mini` | OpenAI model for the agent |
| `MAX_RESULTS` | `10` | Max SQL result rows returned |
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
  "sql": "SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity) ...",
  "session_id": "user-abc123",
  "rewritten_question": "Show top 5 products by revenue as a bar chart",
  "wants_chart": true,
  "chart": {
    "chart_type": "bar",
    "title": "Top 5 Products by Revenue",
    "x_col": "productname",
    "y_col": "revenue",
    "rows": 5,
    "image_base64": "iVBORw0KGgoAAAANS..."
  }
}
```

### `POST /chat`
Text answer + SQL only — no chart generation.

### `GET /history/{session_id}`
Returns full conversation history for a session.

### `DELETE /history/{session_id}`
Clears conversation history for a session.

### `GET /tables`
Lists all available tables in the connected database.

### `GET /health`
Health check — returns model name and version.

---