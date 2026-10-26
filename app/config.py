"""
config.py - Centralized configuration using Pydantic Settings
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()
import os

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB", "inventory_agent")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk--")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # OpenAI
    openai_api_key: str = OPENAI_API_KEY

    # Business DB (Chinook / your data)
    postgres_host: str = POSTGRES_HOST
    postgres_port: int = POSTGRES_PORT
    postgres_user: str = "postgres"
    postgres_password: str = POSTGRES_PASSWORD
    postgres_db: str = "chinook"

    # Agent Memory DB
    memory_postgres_host: str = POSTGRES_HOST
    memory_postgres_port: int = POSTGRES_PORT
    memory_postgres_user: str = "postgres"
    memory_postgres_password: str = POSTGRES_PASSWORD
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

    @property
    def business_db_url(self) -> str:
        """SQLAlchemy URL for the business / Chinook database."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def memory_db_url(self) -> str:
        """SQLAlchemy URL for the conversation memory database."""
        return (
            f"postgresql+psycopg2://{self.memory_postgres_user}:{self.memory_postgres_password}"
            f"@{self.memory_postgres_host}:{self.memory_postgres_port}/{self.memory_postgres_db}"
        )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once at startup)."""
    return Settings()
