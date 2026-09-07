"""Settings. Everything is env-driven so nothing needs editing to deploy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

EmbedderName = Literal["hashing", "sentence-transformers", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YARA_", env_file=".env", extra="ignore"
    )

    # --- Retrieval ---
    embedder: EmbedderName = "hashing"
    embedding_dim: int = 1024
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 8
    rrf_k: int = 60

    # --- Storage ---
    data_dir: Path = Path(os.getenv("YARA_DATA_DIR", ".yara"))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "yara.sqlite3"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "index"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings(**overrides) -> Settings:
    """Process-wide settings singleton. Pass overrides to build a fresh one."""
    global _settings
    if overrides:
        return Settings(**overrides)
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
