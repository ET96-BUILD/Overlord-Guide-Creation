"""API routes — POST /v1/sop."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from sopgen.api.schemas import SOPErrorResponse, SOPResponse
from sopgen.core.config import Settings
from sopgen.core.ffmpeg import FFmpegExtractor
from sopgen.core.jobs import JobManager
from sopgen.core.mime import is_supported_video, supported_types_list, validate_video_file
from sopgen.core.validation import run_with_repair
from sopgen.gemini.client import GeminiClient
from sopgen.gemini.video_analyze import VideoAnalyzer
from sopgen.render.packager import SOPPackager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/sop",
    response_model=SOPResponse,
    responses={
        400: {"model": SOPErrorResponse, "description": "Invalid input"},
        500: {"model": SOPErrorResponse, "description": "Processing failure"},
        502: {"model": SOPErrorResponse, "description": "Gemini API error"},
    },
)
async def generate_sop(
    request: Request,
    video: UploadFile = File(..., description="Screen recording video file"),
    title_hint: Optional[str] = Form(None, description="Suggested SOP title"),
    domain_hint: Optional[str] = Form(None, description='E.g. "NetSuite AP process"'),
    media_resolution: Optional[str] = Form(None, description='"low" or "default"'),
    fps_override: Optional[int] = Form(None, description="Override default 1 FPS sampling"),
) -> SOPResponse:
    """Accept a video upload and return a structured SOP with screenshots."""

    settings: Settings = request.app.state.settings

    # ── Apply per-request overrides ─────────────────────────────────
    if media_resolution in ("low", "default"):
        settings = settings.model_copy(update={"gemini_media_resolution": media_resolution})
    if fps_override is not None:
        settings = settings.model_copy(update={"gemini_video_fps_override": fps_override})

    # ── 1. Save upload ──────────────────────────────────────────────
    jobs = JobManager(settings)
    job_id = jobs.create_job()
    logger.info("[%s] Job created", job_id)

    filename = video.filename or "upload.mp4"
    upload_path = jobs.upload_path(job_id, filename)
    content = await video.read()
    upload_path.write_bytes(content)
    logger.info("[%s] Saved upload (%d bytes) → %s", job_id, len(content), upload_path)

    # ── 2. Validate MIME ────────────────────────────────────────────
    is_valid, mime_type = validate_video_file(upload_path)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {mime_type}. Supported: {supported_types_list()}",
        )

    # ── 3. Gemini analysis + validation/repair ──────────────────────
    try:
        gemini = GeminiClient(settings)
        analyzer = VideoAnalyzer(gemini, settings)

        sop = run_with_repair(
            analyzer,
            upload_path,
            mime_type,
            title_hint=title_hint,
            domain_hint=domain_hint,
            fps_override=fps_override,
            max_retries=settings.max_retry_attempts,
        )
    except ValueError as exc:
        # Validation exhausted
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[%s] Gemini API error", job_id)
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {exc}",
        ) from exc

    # ── 4. Extract screenshots ──────────────────────────────────────
    packager = SOPPackager()
    timestamps = packager.collect_timestamps(sop)

    ffmpeg = FFmpegExtractor(settings.ffmpeg_path)
    frame_map = ffmpeg.extract_all(upload_path, timestamps, jobs.images_dir(job_id))

    # ── 5. Package & return ─────────────────────────────────────────
    package = packager.package(sop, frame_map, job_id)
    packager.save(package, jobs.sop_json_path(job_id))

    logger.info("[%s] Done — %d steps, %d images", job_id, len(sop.steps), len(frame_map))
    return SOPResponse(**package)
