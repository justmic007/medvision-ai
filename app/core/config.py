"""Application configuration.

Centralizes settings so nothing downstream reads os.environ directly.
Values can be overridden via environment variables or a .env file.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "MedVision AI"
    app_version: str = "0.1.0"
    # Non-diagnostic positioning is a project-wide invariant (see DECISIONS.md D-04, D-05).
    disclaimer: str = (
        "MedVision AI is a research/educational prototype and is NOT a "
        "diagnostic tool. Outputs are not a substitute for evaluation by a "
        "qualified clinician."
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
