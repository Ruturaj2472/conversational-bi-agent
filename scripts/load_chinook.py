"""
scripts/load_chinook.py

Downloads the Chinook SQLite database and migrates it to PostgreSQL.
Run this once before starting the agent.

Usage:
    python scripts/load_chinook.py
"""

import os
import sys
import pathlib
import sqlite3
import requests
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
SQLITE_PATH = pathlib.Path("Chinook.db")

# Read from environment (or set defaults for local dev)
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
PG_DB = os.getenv("POSTGRES_DB", "chinook")

PG_URL = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"


def download_chinook():
    if SQLITE_PATH.exists():
        logger.info(f"✓ {SQLITE_PATH} already exists, skipping download.")
        return
    logger.info(f"Downloading Chinook database from {CHINOOK_URL}...")
    response = requests.get(CHINOOK_URL, timeout=30)
    response.raise_for_status()
    SQLITE_PATH.write_bytes(response.content)
    logger.info(f"✓ Downloaded and saved to {SQLITE_PATH}")


def sqlite_type_to_pg(sqlite_type: str) -> str:
    """Map SQLite types to PostgreSQL types."""
    t = sqlite_type.upper()
    if "INT" in t:
        return "INTEGER"
    if "FLOAT" in t or "REAL" in t or "DOUBLE" in t or "NUMERIC" in t:
        return "NUMERIC(10,2)"
    if "NVARCHAR" in t or "VARCHAR" in t or "CHAR" in t or "TEXT" in t:
        # Extract length if present: NVARCHAR(200) -> VARCHAR(200)
        import re

        m = re.search(r"\((\d+)\)", t)
        length = m.group(1) if m else "500"
        return f"VARCHAR({length})"
    if "DATE" in t or "TIME" in t:
        return "TIMESTAMP"
    return "TEXT"


def migrate_to_postgres():
    logger.info(f"Connecting to SQLite: {SQLITE_PATH}")
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    sqlite_cursor = sqlite_conn.cursor()

    logger.info(f"Connecting to PostgreSQL: {PG_URL}")
    pg_engine = create_engine(PG_URL)

    # Get all tables
    sqlite_cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in sqlite_cursor.fetchall()]
    logger.info(f"Tables to migrate: {tables}")

    with pg_engine.begin() as pg_conn:
        for table in tables:
            logger.info(f"  Migrating table: {table}")

            # Get SQLite schema
            sqlite_cursor.execute(f"PRAGMA table_info('{table}')")
            columns = sqlite_cursor.fetchall()
            # columns: (cid, name, type, notnull, dflt_value, pk)

            # Build CREATE TABLE
            col_defs = []
            pk_cols = []
            for cid, name, col_type, notnull, dflt_value, pk in columns:
                pg_type = sqlite_type_to_pg(col_type or "TEXT")
                not_null = "NOT NULL" if notnull else ""
                col_defs.append(f'"{name}" {pg_type} {not_null}'.strip())
                if pk:
                    pk_cols.append(f'"{name}"')

            if pk_cols:
                col_defs.append(f"PRIMARY KEY ({', '.join(pk_cols)})")

            create_sql = (
                f'CREATE TABLE IF NOT EXISTS "{table}" (\n  '
                + ",\n  ".join(col_defs)
                + "\n)"
            )

            pg_conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
            pg_conn.execute(text(create_sql))

            # Migrate data
            sqlite_cursor.execute(f'SELECT * FROM "{table}"')
            rows = sqlite_cursor.fetchall()
            if rows:
                col_names = [f'"{col[1]}"' for col in columns]
                placeholders = ", ".join([f":p{i}" for i in range(len(columns))])
                insert_sql = f'INSERT INTO "{table}" ({", ".join(col_names)}) VALUES ({placeholders})'
                for row in rows:
                    params = {f"p{i}": val for i, val in enumerate(row)}
                    pg_conn.execute(text(insert_sql), params)

            logger.info(f"    ✓ {len(rows)} rows migrated")

    sqlite_conn.close()
    logger.info("✅ Chinook database migrated to PostgreSQL successfully!")


if __name__ == "__main__":
    try:
        download_chinook()
        migrate_to_postgres()
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)
