"""Job lifecycle: ID generation and directory layout."""

from __future__ import annotations

import uuid
from pathlib import Path

from sopgen.core.config import Settings


class JobManager:
    """Creates job IDs and resolves paths for artifacts."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create_job(self) -> str:
        """Return a new unique job ID and ensure its directory tree exists."""
        job_id = uuid.uuid4().hex[:12]
        self._ensure_dirs(job_id)
        return job_id

    # ── path helpers ────────────────────────────────────────────────────

    def job_dir(self, job_id: str) -> Path:
        return self.settings.jobs_dir / job_id

    def upload_path(self, job_id: str, original_filename: str) -> Path:
        suffix = Path(original_filename).suffix or ".mp4"
        return self.settings.uploads_dir / f"{job_id}{suffix}"

    def images_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "images"

    def sop_json_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "sop.json"

    # ── internals ───────────────────────────────────────────────────────

    def _ensure_dirs(self, job_id: str) -> None:
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        self.images_dir(job_id).mkdir(parents=True, exist_ok=True)
