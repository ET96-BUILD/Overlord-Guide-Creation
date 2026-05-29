"""Centralized configuration via environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All runtime configuration.  Loaded from env vars prefixed ``SOPGEN_``."""

    # ── Gemini API ──────────────────────────────────────────────────────
    gemini_api_key: str = ""
    # gemini-2.0-flash was retired by Google for new users on 2026-05-08;
    # 2.5-pro is the current production default.
    gemini_model: str = "gemini-2.5-pro"
    # Wall-clock timeout for a single Gemini request (covers long-running
    # video inference). Default 10 minutes; google-genai's HttpOptions
    # takes ms internally — we convert at the client.
    gemini_request_timeout_seconds: int = 600
    # Transparent retry on 503 UNAVAILABLE (Gemini's "the requested model
    # is over-demand right now" response). Backoff is base * 3^attempt,
    # so defaults give 3 attempts with 5s + 15s = 20s total wait worst
    # case. Other errors (400 / 429 / 500 / network) surface immediately.
    gemini_unavailable_max_attempts: int = 3
    gemini_unavailable_base_delay_seconds: int = 5

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
    # Persistent guides-created counter. Different default dir than data_dir
    # so a Cloud Run gcsfuse mount can target it without overlaying job data.
    stats_path: Path = Path("var/stats.json")

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
