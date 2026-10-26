"""
scripts/load_northwind.py

Downloads the Northwind database SQL file and loads it into PostgreSQL.
Everything lands in the public schema — no prefixes, no multi-schema issues.

Tables created:
    categories, customers, employees, shippers, suppliers,
    products, orders, order_details, employee_territories, territories, regions

Usage:
    python scripts/load_northwind.py

Requirements:
    pip install requests psycopg2-binary python-dotenv
"""

import logging
import os
import sys

import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
PG_DB = os.getenv("POSTGRES_DB", "northwind")

# Clean PostgreSQL port of Northwind — single .sql file, public schema, no issues
NORTHWIND_SQL_URL = (
    "https://raw.githubusercontent.com/pthom/northwind_psql/master/northwind.sql"
)


def download_sql() -> str:
    """Download the Northwind SQL file and return its contents as a string."""
    logger.info("Downloading Northwind SQL from GitHub...")
    r = requests.get(NORTHWIND_SQL_URL, timeout=30)
    r.raise_for_status()
    logger.info(f"  ✓ Downloaded ({len(r.content) // 1024} KB)")
    return r.text


def create_database() -> None:
    """Create the target database if it doesn't exist."""
    # Connect to the default postgres DB to create the target DB
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname="postgres",
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = %s", (PG_DB,))
    exists = cur.fetchone()
    if exists:
        logger.info(f"  ✓ Database '{PG_DB}' already exists.")
    else:
        cur.execute(f'CREATE DATABASE "{PG_DB}"')
        logger.info(f"  ✓ Database '{PG_DB}' created.")

    cur.close()
    conn.close()


def load_sql(sql: str) -> None:
    """Execute the Northwind SQL against the target database."""
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
    )
    conn.autocommit = False
    cur = conn.cursor()

    logger.info(f"Loading Northwind into '{PG_DB}'...")
    cur.execute(sql)
    conn.commit()
    logger.info("  ✓ All statements executed.")

    # Print table summary using information_schema (always available immediately)
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    tables = [row[0] for row in cur.fetchall()]
    logger.info("")
    logger.info("Tables created:")
    for t in tables:
        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
        count = cur.fetchone()[0]
        logger.info(f"  {t:<30} {count:>6} rows")

    cur.close()
    conn.close()


def verify() -> None:
    """Quick row-count check on key tables."""
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB,
    )
    cur = conn.cursor()

    checks = ["products", "orders", "order_details", "suppliers", "customers"]
    logger.info("")
    logger.info("Verification:")
    all_ok = True
    for table in checks:
        try:
            cur.execute(f'SELECT COUNT(*) FROM "{table}"')
            count = cur.fetchone()[0]
            status = "✓" if count > 0 else "⚠ EMPTY"
            logger.info(f"  {status}  {table:<20} {count} rows")
            if count == 0:
                all_ok = False
        except Exception as exc:
            logger.warning(f"  ✗  {table}: {exc}")
            all_ok = False

    cur.close()
    conn.close()

    if all_ok:
        logger.info("")
        logger.info("✅ Northwind loaded successfully!")
        logger.info(f"   Update your .env: POSTGRES_DB={PG_DB}")
    else:
        logger.warning("⚠  Some tables may be empty — check the output above.")


if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  Northwind → PostgreSQL loader")
    logger.info("=" * 55)
    logger.info(f"  Target: {PG_DB} on {PG_HOST}:{PG_PORT}")
    logger.info("")

    try:
        create_database()
        sql = download_sql()
        load_sql(sql)
        verify()
    except psycopg2.OperationalError as exc:
        logger.error(f"Cannot connect to PostgreSQL: {exc}")
        logger.error("Check your .env credentials and make sure Postgres is running.")
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        logger.error(f"Download failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Load failed: {exc}", exc_info=True)
        sys.exit(1)
