"""Central configuration for YARA.

All runtime knobs are environment driven so the same image runs locally,
in CI, and in a container without code changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["anthropic", "gemini", "openai", "echo"]
EmbedderName = Literal["hashing", "sentence-transformers", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YARA_", env_file=".env", extra="ignore"
    )

    # --- LLM ---
    provider: ProviderName = "echo"
    model: str = "claude-sonnet-4-5"
    temperature: float = 0.2
    max_tokens: int = 2048
    request_timeout: float = 90.0
    max_retries: int = 3

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    # --- Retrieval ---
    embedder: EmbedderName = "hashing"
    embedding_dim: int = 1024
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 8
    rrf_k: int = 60

    # --- Orchestration ---
    max_subquestions: int = 5
    max_tool_iterations: int = 6
    max_revisions: int = 2
    min_citation_coverage: float = 0.8

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
