"""
Централизованная конфигурация NGT Memory API.

Все переменные окружения валидируются при старте через Pydantic BaseSettings.
Если обязательная переменная отсутствует — приложение упадёт с понятной ошибкой.

Использование:
    from api.config import settings

    print(settings.chat_model)
    print(settings.openai_api_key.get_secret_value())
"""

from typing import List

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Конфигурация NGT Memory API из переменных окружения."""

    # ── OpenAI ───────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(
        ...,
        description="OpenAI API key (обязательно)",
        alias="OPENAI_API_KEY",
    )

    # ── Models ───────────────────────────────────────────────────────────
    chat_model: str = Field(
        default="gpt-4.1-nano",
        description="Chat модель OpenAI",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding модель OpenAI",
    )

    # ── Memory ───────────────────────────────────────────────────────────
    memory_top_k: int = Field(default=5, ge=1, le=50, description="Top-K воспоминаний для контекста")
    memory_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Минимальный cosine score")
    use_graph: bool = Field(default=True, description="Использовать graph retrieval")

    # ── Sessions ─────────────────────────────────────────────────────────
    session_ttl: int = Field(default=3600, ge=60, description="TTL сессии в секундах")
    max_sessions: int = Field(default=100, ge=1, description="Максимум активных сессий")

    # ── Server ──────────────────────────────────────────────────────────
    root_path: str = Field(default="", description="Root path prefix (e.g. /api for reverse proxy)")

    # ── Billing ────────────────────────────────────────────────────────
    billing_enabled: bool = Field(default=False, description="Включить billing модуль (API-ключи, оплата)")

    # ── Database (для billing) ─────────────────────────────────────────
    db_host: str = Field(default="localhost", description="PostgreSQL host")
    db_port: int = Field(default=5432, description="PostgreSQL port")
    db_name: str = Field(default="ngt_billing", description="PostgreSQL database name")
    db_user: str = Field(default="ngt_user", description="PostgreSQL user")
    db_password: SecretStr = Field(default=SecretStr(""), description="PostgreSQL password")

    # ── SMTP (для billing) ─────────────────────────────────────────────
    smtp_host: str = Field(default="", alias="MAIL_HOST", description="SMTP host")
    smtp_port: int = Field(default=465, alias="MAIL_PORT", description="SMTP port")
    smtp_user: str = Field(default="", alias="MAIL_USERNAME", description="SMTP username")
    smtp_password: SecretStr = Field(default=SecretStr(""), alias="MAIL_PASSWORD", description="SMTP password")
    smtp_from: str = Field(default="", alias="MAIL_FROM_ADDRESS", description="SMTP from address")
    smtp_ssl: bool = Field(default=True, description="Use SSL for SMTP (port 465)")

    # ── ЮKassa (для billing) ──────────────────────────────────────────
    yukassa_shop_id: str = Field(default="", description="ЮKassa Shop ID")
    yukassa_secret_key: SecretStr = Field(default=SecretStr(""), description="ЮKassa Secret Key")
    yukassa_return_url: str = Field(default="https://ngt-memory.ru/payment/success", description="ЮKassa return URL")

    # ── Auth ────────────────────────────────────────────────────────────
    api_secret: str = Field(default="", description="Опциональный secret для защиты endpoint'ов")

    # ── CORS ─────────────────────────────────────────────────────────────
    cors_origins: str = Field(default="*", description="CORS origins через запятую")

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Уровень логирования")
    log_json: bool = Field(default=True, description="JSON формат логов")

    # ── Meta ─────────────────────────────────────────────────────────────
    version: str = Field(default="0.23.0", description="Версия API")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    model_config = {
        "env_prefix": "NGT_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
        "populate_by_name": True,
    }


# Единственный экземпляр — импортировать отовсюду
settings = Settings()
