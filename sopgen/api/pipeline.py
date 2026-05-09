"""Async runner for the SOP pipeline.

Wraps the existing blocking pipeline (Gemini analysis + ffmpeg + packaging)
in ``asyncio.to_thread`` so the FastAPI event loop isn't blocked, and
publishes per-stage updates to the ``JobRegistry`` so clients can poll
progress.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from sopgen.api.job_registry import (
    JobRegistry,
    STAGE_ANALYZING,
    STAGE_DONE,
    STAGE_EXTRACTING,
    STAGE_PACKAGING,
    STATUS_DONE,
    STATUS_RUNNING,
)
from sopgen.api.stats import GuidesStats
from sopgen.core.config import Settings
from sopgen.core.ffmpeg import FFmpegExtractor
from sopgen.core.jobs import JobManager
from sopgen.core.validation import run_with_repair
from sopgen.gemini.client import GeminiClient
from sopgen.gemini.video_analyze import VideoAnalyzer
from sopgen.render.docx_packager import write_docx
from sopgen.render.packager import SOPPackager
from sopgen.render.zip_packager import BUNDLE_FILENAME, write_zip

logger = logging.getLogger(__name__)


# ── Public API ──────────────────────────────────────────────────────────


async def run_pipeline_async(
    *,
    registry: JobRegistry,
    settings: Settings,
    job_id: str,
    upload_path: Path,
    mime_type: str,
    title_hint: Optional[str],
    domain_hint: Optional[str],
    fps_override: Optional[int],
    stats: Optional[GuidesStats] = None,
    user_email: Optional[str] = None,
) -> None:
    """Run the SOP pipeline for *job_id*, updating *registry* as it goes.

    Caught exceptions are recorded on the registry as ``status=error``.
    Never re-raises — this is fire-and-forget from the route handler's
    perspective.

    On a successful completion (and only on success), the *stats* counter
    is incremented exactly once. Errors do not bump the counter.
    """
    jobs = JobManager(settings)
    try:
        registry.set_stage(job_id, STAGE_ANALYZING, status=STATUS_RUNNING)
        sop = await asyncio.to_thread(
            _run_gemini,
            settings,
            upload_path,
            mime_type,
            title_hint,
            domain_hint,
            fps_override,
        )

        registry.set_stage(job_id, STAGE_EXTRACTING)
        packager = SOPPackager()
        timestamps = packager.collect_timestamps(sop)
        ffmpeg = FFmpegExtractor(settings.ffmpeg_path)
        frame_map = await asyncio.to_thread(
            ffmpeg.extract_all,
            upload_path,
            timestamps,
            jobs.images_dir(job_id),
        )

        registry.set_stage(job_id, STAGE_PACKAGING)
        await asyncio.to_thread(
            _finalize_outputs, sop, frame_map, packager, jobs, job_id
        )

        registry.set_stage(job_id, STAGE_DONE, status=STATUS_DONE)
        # Only successful runs bump the guides-created counter. The
        # increment is intentionally inside the try-block AFTER the
        # status flip, so any failure during file write is reported via
        # set_error (counter stays at the prior value).
        if stats is not None:
            new_total = stats.increment(email=user_email)
            logger.info(
                "[%s] Pipeline done — guides_created=%d (user=%s)",
                job_id,
                new_total,
                user_email or "anon",
            )
        else:
            logger.info("[%s] Pipeline done", job_id)
    except Exception as exc:  # noqa: BLE001 — anything is a job failure
        logger.exception("[%s] Pipeline failed", job_id)
        registry.set_error(job_id, str(exc))


# ── Blocking helpers (run in worker threads) ────────────────────────────


def _run_gemini(
    settings: Settings,
    upload_path: Path,
    mime_type: str,
    title_hint: Optional[str],
    domain_hint: Optional[str],
    fps_override: Optional[int],
):
    gemini = GeminiClient(settings)
    analyzer = VideoAnalyzer(gemini, settings)
    return run_with_repair(
        analyzer,
        upload_path,
        mime_type,
        title_hint=title_hint,
        domain_hint=domain_hint,
        fps_override=fps_override,
        max_retries=settings.max_retry_attempts,
    )


def _finalize_outputs(
    sop,
    frame_map: dict,
    packager: SOPPackager,
    jobs: JobManager,
    job_id: str,
) -> None:
    """Write sop.json + sop.docx + sop_bundle.zip for *job_id*."""
    package = packager.package(
        sop,
        frame_map,
        job_id,
        output_images_dir=jobs.images_dir(job_id),
    )
    packager.save(package, jobs.sop_json_path(job_id))
    write_docx(package["sop"], jobs.job_dir(job_id) / "sop.docx")
    write_zip(jobs.job_dir(job_id), jobs.job_dir(job_id) / BUNDLE_FILENAME)
