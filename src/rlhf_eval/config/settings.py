from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings, loaded from environment variables and .env files.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RLHF_",        
    )

    database_url: str
    database_pool_size: int = Field(default=5, ge=1, le=20)
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    @field_validator("log_level")
    @classmethod
    def validate_log_levels(cls, v: str) -> str:

        upper_v = v.upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. "
                             f"Must be one of the following: {valid_levels}")
        return upper_v

@lru_cache
def get_settings() -> Settings:
    return Settings()
    
