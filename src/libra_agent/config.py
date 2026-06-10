"""환경변수 → 설정 매핑. pydantic-settings."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Checkpoint database (LangGraph runtime state only) ---
    database_url: str = "postgresql://libra:libra@localhost:5432/libra"

    # --- Anthropic ---
    anthropic_api_key: str = Field(default="")
    anthropic_haiku_model: str = "claude-haiku-4-5"
    anthropic_sonnet_model: str = "claude-haiku-4-5"

    # --- S3 ---
    s3_bucket: str = ""
    aws_region: str = "ap-northeast-2"

    # --- Knowledge cache / ingest worker handoff ---
    knowledge_cache_dir: str = "/opt/libra/knowledge/current"
    knowledge_s3_prefix: str = "knowledge/current"
    ingest_jobs_enabled: bool = False

    # --- Logging ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "console"

    # --- CORS ---
    # 정상 흐름은 Spring 만 호출하지만 dev 시 Vue 직접 호출도 허용
    allowed_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:8080",
    ]


settings = Settings()
