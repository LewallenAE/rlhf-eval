#!/usr/bin/env python3
"""
 Enter module docstring here
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
from functools import lru_cache
from typing import Final, FrozenSet

# ----------------- Third Party Library -----------------
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ----------------- Application Imports -----------------


# ----------------- Module-level Configuration -----------------

VALID_LOG_LEVELS: Final[FrozenSet[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)
class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RLHF_",
        extra="ignore",  # explicit: ignore unexpected env keys
    )

    database_url: str
    database_pool_size: int = Field(default=5, ge=1, le=20)
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: object) -> str:
        # Explicit normalization: accept strings with whitespace, accept non-strings via str()
        value = str(v).strip().upper()

        if value not in VALID_LOG_LEVELS:
            allowed = ", ".join(sorted(VALID_LOG_LEVELS))
            raise ValueError(f"Invalid log_level={v!r}. Must be one of: {allowed}.")

        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
