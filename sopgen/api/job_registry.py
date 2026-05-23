"""In-memory job registry for the async SOP pipeline.

Process-local on purpose for v1 — a single uvicorn worker can hold this
state without a backing store. When the service grows past one instance,
swap this for Redis (or similar) without changing call sites: the
``JobRegistry`` API is intentionally minimal.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Allowed status / stage values ───────────────────────────────────────

# These are surfaced to clients via /v1/jobs/<id>/status. Stages are free
# text per spec; we keep a canonical list here so the frontend has a
# fixed vocabulary to map into user-facing labels.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

STAGE_QUEUED = "queued"
STAGE_UPLOADING = "uploading"
STAGE_TRANSCODING = "transcoding"
STAGE_ANALYZING = "analyzing"
STAGE_EXTRACTING = "extracting frames"
STAGE_PACKAGING = "packaging"
STAGE_DONE = "done"
STAGE_ERROR = "error"


@dataclass
class JobEntry:
    job_id: str
    output_dir: Path
    status: str = STATUS_QUEUED
    stage: str = STAGE_QUEUED
    error: Optional[str] = None


class JobRegistry:
    """Thread-safe registry of in-flight and completed jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobEntry] = {}
        self._lock = threading.Lock()
        # Hold strong refs to background tasks so they aren't GC'd before
        # they finish. Tasks self-discard via add_done_callback.
        self._tasks: set[asyncio.Task] = set()

    # ── Mutators ────────────────────────────────────────────────────────

    def create(self, job_id: str, output_dir: Path) -> JobEntry:
        entry = JobEntry(job_id=job_id, output_dir=output_dir)
        with self._lock:
            self._jobs[job_id] = entry
        return entry

    def set_stage(
        self,
        job_id: str,
        stage: str,
        *,
        status: Optional[str] = None,
    ) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                return
            entry.stage = stage
            if status is not None:
                entry.status = status

    def set_error(self, job_id: str, message: str) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                return
            entry.status = STATUS_ERROR
            entry.stage = STAGE_ERROR
            entry.error = message

    def track_task(self, task: asyncio.Task) -> None:
        """Hold a strong reference to *task* until it completes."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ── Readers ─────────────────────────────────────────────────────────

    def get(self, job_id: str) -> Optional[JobEntry]:
        with self._lock:
            return self._jobs.get(job_id)
