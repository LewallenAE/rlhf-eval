#!/usr/bin/env python3
"""
Application and pipeline configuration loaded from environment variables.
"""

# ----------------- Futures -----------------
from __future__ import annotations

# ----------------- Standard Library -----------------
from functools import lru_cache
from typing import Any, Final, FrozenSet

# ----------------- Third Party Library -----------------
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ----------------- Application Imports -----------------


# ----------------- Module-level Configuration -----------------

VALID_LOG_LEVELS: Final[FrozenSet[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


class DetectorConfig(BaseModel):
    """Configuration for a single quality detector."""

    enabled: bool = True
    threshold_percentile: float = Field(default=99.5, ge=0.0, le=100.0)
    extra: dict[str, Any] = Field(default_factory=dict)


# Default detector configurations
DEFAULT_DETECTORS: dict[str, DetectorConfig] = {
    "semantic_similarity": DetectorConfig(threshold_percentile=99.5),
    "readability_mismatch": DetectorConfig(threshold_percentile=95.0),
    "repetition": DetectorConfig(threshold_percentile=1.0),  # lower is worse → low percentile
    "length_ratio": DetectorConfig(threshold_percentile=5.0),  # lower is worse → low percentile
    "refusal_bias": DetectorConfig(threshold_percentile=99.0),
    "unsafe_prompt": DetectorConfig(threshold_percentile=99.0),
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RLHF_",
        extra="ignore",  # explicit: ignore unexpected env keys
    )

    # Database
    database_url: str
    database_pool_size: int = Field(default=5, ge=1, le=20)

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    # General
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Pipeline processing
    batch_size: int = Field(default=32, ge=1)
    block_size: int = Field(default=1000, ge=1)

    # Models
    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # Detector configs (populated via get_detector_configs)
    detectors: dict[str, DetectorConfig] = Field(
        default_factory=lambda: dict(DEFAULT_DETECTORS)
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: object) -> str:
        # Explicit normalization: accept strings with whitespace, accept non-strings via str()
        value = str(v).strip().upper()

        if value not in VALID_LOG_LEVELS:
            allowed = ", ".join(sorted(VALID_LOG_LEVELS))
            raise ValueError(f"Invalid log_level={v!r}. Must be one of: {allowed}.")

        return value

    def get_detector_config(self, detector_name: str) -> DetectorConfig:
        """Get config for a specific detector, falling back to defaults."""
        return self.detectors.get(detector_name, DetectorConfig())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
