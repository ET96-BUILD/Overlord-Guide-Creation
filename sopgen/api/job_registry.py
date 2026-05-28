"""Disk-backed job registry for the async SOP pipeline.

Keeps an in-memory dict for the hot path and writes each status update
to ``<job_dir>/status.json`` so a container restart (or a request that
lands on a freshly-booted Cloud Run instance) can still recover the
job's state from disk.

Process-local in-memory writes are still authoritative for the running
process; the disk file is a durable mirror. Cross-instance writes from
different containers will race on the same file — fine for the current
``--max-instances=1`` Cloud Run setup, not safe at horizontal scale.
At that point swap this for Firestore or Redis without touching call
sites: the ``JobRegistry`` API is intentionally minimal.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sopgen.core.jobs import JobManager

logger = logging.getLogger(__name__)

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

_STATUS_FILENAME = "status.json"


@dataclass
class JobEntry:
    job_id: str
    output_dir: Path
    status: str = STATUS_QUEUED
    stage: str = STAGE_QUEUED
    error: Optional[str] = None


# ── Serializer / parser ─────────────────────────────────────────────────


def _entry_to_disk_dict(entry: JobEntry) -> dict:
    """Schema: {status, stage, error, updated_at}. output_dir is implied
    by job_id + JobManager and therefore not serialized."""
    return {
        "status": entry.status,
        "stage": entry.stage,
        "error": entry.error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _entry_from_disk_dict(
    job_id: str, output_dir: Path, data: dict
) -> Optional[JobEntry]:
    """Reconstruct a JobEntry from the on-disk JSON dict. Returns None
    if the dict is missing required fields — the caller treats that as
    a cache miss rather than raising."""
    if not isinstance(data, dict):
        return None
    if "status" not in data or "stage" not in data:
        return None
    try:
        return JobEntry(
            job_id=job_id,
            output_dir=output_dir,
            status=str(data["status"]),
            stage=str(data["stage"]),
            error=data.get("error"),
        )
    except (TypeError, ValueError):
        return None


# ── Registry ────────────────────────────────────────────────────────────


class JobRegistry:
    """Thread-safe registry of in-flight and completed jobs."""

    def __init__(self, jobs: Optional[JobManager] = None) -> None:
        self._jobs: dict[str, JobEntry] = {}
        self._lock = threading.Lock()
        # Hold strong refs to background tasks so they aren't GC'd before
        # they finish. Tasks self-discard via add_done_callback. Strictly
        # in-memory — asyncio.Task can't be persisted across instances.
        self._tasks: set[asyncio.Task] = set()
        # JobManager resolves per-job dirs for disk persistence. When
        # None, the registry runs in pure-in-memory mode (used by some
        # tests + as a safety fallback if init wiring is incomplete).
        self._jobs_mgr = jobs

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
            self._persist_locked(entry)

    def set_error(self, job_id: str, message: str) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                return
            entry.status = STATUS_ERROR
            entry.stage = STAGE_ERROR
            entry.error = message
            self._persist_locked(entry)

    def track_task(self, task: asyncio.Task) -> None:
        """Hold a strong reference to *task* until it completes."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # ── Readers ─────────────────────────────────────────────────────────

    def get(self, job_id: str) -> Optional[JobEntry]:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is not None:
                return entry
            # In-memory cache miss — try the disk fallback (used after a
            # container restart, or by a different Cloud Run instance).
            entry = self._read_from_disk_locked(job_id)
            if entry is not None:
                self._jobs[job_id] = entry  # cache for subsequent lookups
            return entry

    # ── Disk persistence ────────────────────────────────────────────────

    def _persist_locked(self, entry: JobEntry) -> None:
        """Write *entry* to ``<job_dir>/status.json`` atomically. Called
        with self._lock held. Disk failures are logged but never
        propagate — the in-memory state remains authoritative for the
        current request."""
        if self._jobs_mgr is None:
            return
        status_path = self._jobs_mgr.job_dir(entry.job_id) / _STATUS_FILENAME
        payload = _entry_to_disk_dict(entry)
        try:
            self._atomic_write_json(status_path, payload)
        except Exception:
            logger.exception(
                "Failed to persist %s for job %s", _STATUS_FILENAME, entry.job_id
            )

    def _read_from_disk_locked(self, job_id: str) -> Optional[JobEntry]:
        """Read ``<job_dir>/status.json`` if it exists. Called with
        self._lock held. Missing / corrupt → None (caller becomes 404)."""
        if self._jobs_mgr is None:
            return None
        job_dir = self._jobs_mgr.job_dir(job_id)
        status_path = job_dir / _STATUS_FILENAME
        if not status_path.exists():
            return None
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Corrupt or unreadable %s at %s — treating as cache miss",
                _STATUS_FILENAME, status_path,
            )
            return None
        return _entry_from_disk_dict(job_id, job_dir, data)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        """mkstemp + os.replace so a crash mid-write can't leave a torn
        file that then gets read back as a corrupt status."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=".status_", suffix=".json.tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
