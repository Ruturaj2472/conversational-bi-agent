import pytest
from pydantic import ValidationError

from app.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "MEMORY_POSTGRES_HOST",
        "MEMORY_POSTGRES_PORT",
        "MEMORY_POSTGRES_USER",
        "MEMORY_POSTGRES_PASSWORD",
        "MEMORY_POSTGRES_DB",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def make_settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_defaults_load_without_port_env():
    s = make_settings()
    assert s.postgres_port == 5432
    assert s.business_db_url == (
        "postgresql+psycopg2://postgres:postgres@localhost:5432/northwind"
    )


def test_memory_db_falls_back_to_business_db():
    s = make_settings(postgres_host="db.local", postgres_password="secret")
    assert s.memory_postgres_host == "db.local"
    assert s.memory_postgres_password == "secret"
    assert s.memory_db_url.endswith("@db.local:5432/agent_memory")


def test_memory_db_values_override_business_db():
    s = make_settings(memory_postgres_host="mem.local", memory_postgres_port=6543)
    assert s.memory_db_url.endswith("@mem.local:6543/agent_memory")


def test_special_characters_in_password_are_escaped():
    s = make_settings(postgres_password="p@ss:word/1")
    assert "p%40ss%3Aword%2F1@localhost" in s.business_db_url


def test_missing_openai_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValidationError):
        make_settings()
