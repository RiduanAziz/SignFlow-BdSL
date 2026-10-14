from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment and .env files."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_name: str = "sign-language-recognition"
    log_level: str = "INFO"
    seed: int = 42
    device: str = "auto"
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"
    mlflow_tracking_uri: str = "file://./mlruns"
    batch_size: int = 32
    sequence_length: int = 32
    num_classes: int = 100

    @property
    def root_dir(self) -> Path:
        return Path(__file__).resolve().parents[3]


def get_settings() -> Settings:
    return Settings()
