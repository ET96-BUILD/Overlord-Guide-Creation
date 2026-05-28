"""Tests for the disk-backed JobRegistry.

Covers the in-memory hot path, the on-disk mirror that survives a
container restart, and the fail-soft handling of missing / corrupt
status.json files.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from sopgen.api.job_registry import (
    JobRegistry,
    STAGE_ANALYZING,
    STAGE_PACKAGING,
    STATUS_ERROR,
    STATUS_RUNNING,
)
from sopgen.core.config import Settings
from sopgen.core.jobs import JobManager


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def settings(tmp_path):
    return Settings(
        gemini_api_key="test-key",
        data_dir=tmp_path / "data",
    )


@pytest.fixture()
def jobs(settings):
    return JobManager(settings)


@pytest.fixture()
def registry(jobs):
    return JobRegistry(jobs=jobs)


def _seed_job(registry: JobRegistry, jobs: JobManager) -> str:
    """Create a fresh job (registry entry + job_dir on disk) and return its id."""
    job_id = jobs.create_job()
    registry.create(job_id, output_dir=jobs.job_dir(job_id))
    return job_id


# ── set_stage writes status.json ────────────────────────────────────────


class TestSetStagePersistence:
    def test_writes_status_json_with_expected_fields(self, registry, jobs):
        job_id = _seed_job(registry, jobs)
        registry.set_stage(job_id, STAGE_ANALYZING, status=STATUS_RUNNING)

        status_path = jobs.job_dir(job_id) / "status.json"
        assert status_path.exists()
        data = json.loads(status_path.read_text(encoding="utf-8"))

        assert data["status"] == STATUS_RUNNING
        assert data["stage"] == STAGE_ANALYZING
        assert data["error"] is None
        # updated_at must be a parseable ISO-8601 string (with timezone).
        parsed = datetime.fromisoformat(data["updated_at"])
        assert parsed.tzinfo is not None

    def test_subsequent_set_stage_overwrites_file(self, registry, jobs):
        job_id = _seed_job(registry, jobs)
        registry.set_stage(job_id, STAGE_ANALYZING, status=STATUS_RUNNING)
        registry.set_stage(job_id, STAGE_PACKAGING)

        data = json.loads(
            (jobs.job_dir(job_id) / "status.json").read_text(encoding="utf-8")
        )
        # Latest write wins — stage advanced, status preserved.
        assert data["stage"] == STAGE_PACKAGING
        assert data["status"] == STATUS_RUNNING


class TestSetErrorPersistence:
    def test_writes_error_to_disk(self, registry, jobs):
        job_id = _seed_job(registry, jobs)
        registry.set_error(job_id, "simulated boom")

        data = json.loads(
            (jobs.job_dir(job_id) / "status.json").read_text(encoding="utf-8")
        )
        assert data["status"] == STATUS_ERROR
        assert data["stage"] == "error"
        assert data["error"] == "simulated boom"


# ── get() hot path + cold-cache disk fallback ──────────────────────────


class TestGetInMemoryHotPath:
    def test_returns_in_memory_entry_when_present(self, registry, jobs):
        job_id = _seed_job(registry, jobs)
        registry.set_stage(job_id, STAGE_ANALYZING, status=STATUS_RUNNING)

        entry = registry.get(job_id)
        assert entry is not None
        assert entry.job_id == job_id
        assert entry.status == STATUS_RUNNING
        assert entry.stage == STAGE_ANALYZING


class TestGetDiskFallback:
    def test_fresh_registry_reads_state_from_disk(self, settings, jobs):
        """Simulate a container restart: write status via one registry,
        then construct a brand-new registry pointing at the same
        data_dir. get() must reconstruct the entry from disk."""
        original = JobRegistry(jobs=jobs)
        job_id = _seed_job(original, jobs)
        original.set_stage(job_id, STAGE_PACKAGING, status=STATUS_RUNNING)

        # New process — empty in-memory cache, same on-disk state.
        restarted = JobRegistry(jobs=JobManager(settings))
        entry = restarted.get(job_id)

        assert entry is not None
        assert entry.job_id == job_id
        assert entry.status == STATUS_RUNNING
        assert entry.stage == STAGE_PACKAGING
        assert entry.error is None
        assert entry.output_dir == jobs.job_dir(job_id)

    def test_disk_fallback_caches_entry(self, settings, jobs):
        """After the first disk-fallback hit, subsequent get() calls
        should use the in-memory cache (verified by deleting the file
        and confirming get() still returns the entry)."""
        original = JobRegistry(jobs=jobs)
        job_id = _seed_job(original, jobs)
        original.set_stage(job_id, STAGE_ANALYZING, status=STATUS_RUNNING)

        restarted = JobRegistry(jobs=JobManager(settings))
        first = restarted.get(job_id)
        assert first is not None

        # Yank the file out from under the registry; the cached entry
        # should still resolve.
        (jobs.job_dir(job_id) / "status.json").unlink()
        second = restarted.get(job_id)
        assert second is not None
        assert second.stage == STAGE_ANALYZING


class TestGetMissingOrCorrupt:
    def test_returns_none_for_unknown_job(self, registry):
        assert registry.get("no-such-job-id") is None

    def test_returns_none_when_status_json_missing(self, settings, jobs):
        """A job dir that exists but has no status.json (e.g. created
        but never had a stage transition recorded) → cache miss."""
        job_id = jobs.create_job()
        # Note: no registry.create() and no status.json file
        fresh = JobRegistry(jobs=JobManager(settings))
        assert fresh.get(job_id) is None

    def test_returns_none_when_status_json_is_corrupt(self, settings, jobs):
        job_id = jobs.create_job()
        # Hand-write a broken file
        (jobs.job_dir(job_id) / "status.json").write_text(
            "not json at all {", encoding="utf-8"
        )

        fresh = JobRegistry(jobs=JobManager(settings))
        assert fresh.get(job_id) is None

    def test_returns_none_when_status_json_missing_required_fields(
        self, settings, jobs
    ):
        job_id = jobs.create_job()
        # Valid JSON but missing status/stage
        (jobs.job_dir(job_id) / "status.json").write_text(
            json.dumps({"updated_at": "2026-05-09T00:00:00+00:00"}),
            encoding="utf-8",
        )
        fresh = JobRegistry(jobs=JobManager(settings))
        assert fresh.get(job_id) is None


# ── Pure in-memory mode (no JobManager) ─────────────────────────────────


class TestPureInMemoryMode:
    def test_no_jobs_manager_means_no_disk_writes(self, tmp_path):
        """A JobRegistry constructed without a JobManager works exactly
        as before — fully in-memory, no status.json on disk."""
        registry = JobRegistry()  # no jobs= arg
        job_id = "test-job"
        registry.create(job_id, output_dir=tmp_path / "nope")
        registry.set_stage(job_id, STAGE_ANALYZING, status=STATUS_RUNNING)

        entry = registry.get(job_id)
        assert entry is not None
        assert entry.stage == STAGE_ANALYZING
        # No file was written (registry had no JobManager to resolve a path).
        assert not (tmp_path / "nope" / "status.json").exists()
