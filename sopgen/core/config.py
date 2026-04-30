"""Centralized configuration via environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All runtime configuration.  Loaded from env vars prefixed ``SOPGEN_``."""

    # ── Gemini API ──────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # ── Media processing ────────────────────────────────────────────────
    gemini_media_resolution: Literal["low", "default"] = "default"
    gemini_video_fps_override: Optional[int] = None

    # ── Validation ──────────────────────────────────────────────────────
    max_retry_attempts: int = 2
    max_substeps_per_step: int = 4
    min_images_per_step: int = 1

    # ── Storage ─────────────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    max_inline_size_mb: int = 20

    # ── Server ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── FFmpeg ──────────────────────────────────────────────────────────
    ffmpeg_path: str = "ffmpeg"

    # ── Derived helpers ─────────────────────────────────────────────────
    @property
    def uploads_dir(self) -> Path:
        p = self.data_dir / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def jobs_dir(self) -> Path:
        p = self.data_dir / "jobs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def max_inline_bytes(self) -> int:
        return self.max_inline_size_mb * 1024 * 1024

    model_config = {"env_prefix": "SOPGEN_", "env_file": ".env", "extra": "ignore"}
