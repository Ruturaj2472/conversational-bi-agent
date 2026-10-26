"""
config.py - Centralized configuration using Pydantic Settings
"""

from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

# Export .env into os.environ so the OpenAI client can read OPENAI_API_KEY
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # OpenAI
    openai_api_key: str

    # Business DB (Chinook / your data)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "chinook"

    # Agent Memory DB (unset fields fall back to the business DB values)
    memory_postgres_host: Optional[str] = None
    memory_postgres_port: Optional[int] = None
    memory_postgres_user: Optional[str] = None
    memory_postgres_password: Optional[str] = None
    memory_postgres_db: str = "agent_memory"

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "bi-agent"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False

    # Agent
    max_results: int = 10
    model_name: str = "gpt-4.1-mini"
    fast_model_name: str = "gpt-4.1-mini"  # question analysis + chart config
    query_timeout_seconds: int = 30
    history_messages: int = 10  # past messages used to resolve follow-ups
    history_max_chars: int = 500  # longer past answers are truncated

    @model_validator(mode="after")
    def _default_memory_db_to_business_db(self) -> "Settings":
        self.memory_postgres_host = self.memory_postgres_host or self.postgres_host
        self.memory_postgres_port = self.memory_postgres_port or self.postgres_port
        self.memory_postgres_user = self.memory_postgres_user or self.postgres_user
        self.memory_postgres_password = (
            self.memory_postgres_password or self.postgres_password
        )
        return self

    @property
    def business_db_url(self) -> str:
        """SQLAlchemy URL for the business / Chinook database."""
        return URL.create(
            "postgresql+psycopg2",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        ).render_as_string(hide_password=False)

    @property
    def memory_db_url(self) -> str:
        """SQLAlchemy URL for the conversation memory database."""
        return URL.create(
            "postgresql+psycopg2",
            username=self.memory_postgres_user,
            password=self.memory_postgres_password,
            host=self.memory_postgres_host,
            port=self.memory_postgres_port,
            database=self.memory_postgres_db,
        ).render_as_string(hide_password=False)


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once at startup)."""
    return Settings()
