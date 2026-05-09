"""API routes — async POST /v1/sop, polling endpoints, zip download."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from sopgen.api.job_registry import (
    JobRegistry,
    STATUS_DONE,
    STATUS_ERROR,
)
from sopgen.api.pipeline import run_pipeline_async
from sopgen.api.schemas import (
    ConfigResponse,
    JobAcceptedResponse,
    JobStatusResponse,
    SOPErrorResponse,
)
from sopgen.api.stats import GuidesStats
from sopgen.core.config import Settings
from sopgen.core.jobs import JobManager
from sopgen.core.mime import supported_types_list, validate_video_file
from sopgen.render.zip_packager import BUNDLE_FILENAME, slugify, write_zip

logger = logging.getLogger(__name__)

router = APIRouter()


# Models the frontend is allowed to request per-call. Keep this short —
# every entry is a contract: the model must be available to all API keys
# and the prompt has been validated against it.
ALLOWED_MODELS: tuple[str, ...] = (
    "gemini-2.5-pro",
    "gemini-2.5-flash",
)

# Google IAP forwards the verified end-user email here, prefixed with
# "accounts.google.com:". When the service runs behind IAP the header is
# trustworthy; in local/dev the header is simply absent.
_IAP_EMAIL_HEADER = "X-Goog-Authenticated-User-Email"
_IAP_EMAIL_PREFIX = "accounts.google.com:"


def _bare_user_email(request: Request) -> Optional[str]:
    """Read the IAP user email header and strip the IAP prefix."""
    raw = request.headers.get(_IAP_EMAIL_HEADER)
    if not raw:
        return None
    return raw[len(_IAP_EMAIL_PREFIX):] if raw.startswith(_IAP_EMAIL_PREFIX) else raw


# ═══════════════════════════════════════════════════════════════════════
#  GET /v1/config — public config snapshot for the frontend
# ═══════════════════════════════════════════════════════════════════════


@router.get("/config", response_model=ConfigResponse)
def get_config(request: Request) -> ConfigResponse:
    """Read-only public config (no secrets).

    Used by the frontend to pre-select the matching model radio so the
    UI default never drifts from the server's configured default.
    """
    settings: Settings = request.app.state.settings
    return ConfigResponse(default_model=settings.gemini_model)


# ═══════════════════════════════════════════════════════════════════════
#  GET /v1/stats — guides-created counter for the frontend badge
# ═══════════════════════════════════════════════════════════════════════


@router.get("/stats")
def get_stats(request: Request) -> dict:
    stats: GuidesStats = request.app.state.stats
    return {"guides_created": stats.read_count()}


# ═══════════════════════════════════════════════════════════════════════
#  GET /v1/leaderboard — top N users by guides created
# ═══════════════════════════════════════════════════════════════════════


@router.get("/leaderboard")
def get_leaderboard(request: Request, limit: int = 5) -> dict:
    # Clamp limit to a sane range so a stray ?limit=99999 can't pull
    # every user record over the wire.
    if limit < 1:
        limit = 1
    if limit > 100:
        limit = 100
    stats: GuidesStats = request.app.state.stats
    return {"top": stats.read_leaderboard(limit=limit)}


# ═══════════════════════════════════════════════════════════════════════
#  POST /v1/sop — accept upload, kick off pipeline, return 202
# ═══════════════════════════════════════════════════════════════════════


@router.post(
    "/sop",
    status_code=202,
    response_model=JobAcceptedResponse,
    responses={
        400: {"model": SOPErrorResponse, "description": "Invalid input"},
    },
)
async def generate_sop(
    request: Request,
    video: UploadFile = File(..., description="Screen recording video file"),
    title_hint: Optional[str] = Form(None, description="Suggested SOP title"),
    domain_hint: Optional[str] = Form(None, description='E.g. "NetSuite AP process"'),
    media_resolution: Optional[str] = Form(None, description='"low" or "default"'),
    fps_override: Optional[int] = Form(None, description="Override default 1 FPS sampling"),
    model: Optional[str] = Form(
        None,
        description=f"Override Gemini model. Allowed: {', '.join(ALLOWED_MODELS)}",
    ),
) -> JobAcceptedResponse:
    """Accept a video upload and dispatch the SOP pipeline asynchronously.

    The blocking work (Gemini analysis, ffmpeg extraction, packaging) runs
    on the executor pool via ``asyncio.to_thread`` so the event loop stays
    responsive for status polls and other requests.
    """
    settings: Settings = request.app.state.settings

    # ── Apply per-request settings overrides ────────────────────────
    if media_resolution in ("low", "default"):
        settings = settings.model_copy(update={"gemini_media_resolution": media_resolution})
    if fps_override is not None:
        settings = settings.model_copy(update={"gemini_video_fps_override": fps_override})
    if model is not None:
        if model not in ALLOWED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported model: {model!r}. "
                    f"Allowed values: {list(ALLOWED_MODELS)}"
                ),
            )
        settings = settings.model_copy(update={"gemini_model": model})

    # ── 1. Save upload synchronously (fast disk write) ──────────────
    jobs = JobManager(settings)
    job_id = jobs.create_job()
    logger.info("[%s] Job created", job_id)

    filename = video.filename or "upload.mp4"
    upload_path = jobs.upload_path(job_id, filename)
    content = await video.read()
    upload_path.write_bytes(content)
    logger.info("[%s] Saved upload (%d bytes) → %s", job_id, len(content), upload_path)

    # ── 2. Validate MIME (still synchronous; cheap) ─────────────────
    is_valid, mime_type = validate_video_file(upload_path)
    if not is_valid:
        # Clean up the freshly-saved upload so we don't leave litter
        # for a job that never enters the registry.
        upload_path.unlink(missing_ok=True)
        shutil.rmtree(jobs.job_dir(job_id), ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {mime_type}. Supported: {supported_types_list()}",
        )

    # ── 3. Register the job and spawn the pipeline ──────────────────
    registry: JobRegistry = request.app.state.job_registry
    registry.create(job_id, output_dir=jobs.job_dir(job_id))

    stats: GuidesStats = request.app.state.stats
    user_email = _bare_user_email(request)
    task = asyncio.create_task(
        run_pipeline_async(
            registry=registry,
            settings=settings,
            job_id=job_id,
            upload_path=upload_path,
            mime_type=mime_type,
            title_hint=title_hint,
            domain_hint=domain_hint,
            fps_override=fps_override,
            stats=stats,
            user_email=user_email,
        )
    )
    registry.track_task(task)

    return JobAcceptedResponse(job_id=job_id, status="queued")


# ═══════════════════════════════════════════════════════════════════════
#  GET /v1/jobs/{job_id}/status — poll progress
# ═══════════════════════════════════════════════════════════════════════


@router.get(
    "/jobs/{job_id}/status",
    response_model=JobStatusResponse,
    responses={404: {"model": SOPErrorResponse, "description": "Unknown job_id"}},
)
def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    registry: JobRegistry = request.app.state.job_registry
    entry = registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    has_zip = (
        entry.output_dir is not None
        and (entry.output_dir / BUNDLE_FILENAME).exists()
    )
    return JobStatusResponse(
        status=entry.status,
        stage=entry.stage,
        error=entry.error,
        has_zip=has_zip,
    )


# ═══════════════════════════════════════════════════════════════════════
#  GET /v1/jobs/{job_id}/result — final SOP JSON
# ═══════════════════════════════════════════════════════════════════════


@router.get(
    "/jobs/{job_id}/result",
    responses={
        404: {"model": SOPErrorResponse, "description": "Unknown job_id"},
        425: {"model": SOPErrorResponse, "description": "Job still in flight"},
        500: {"model": SOPErrorResponse, "description": "Job failed"},
    },
)
def get_job_result(job_id: str, request: Request) -> dict:
    registry: JobRegistry = request.app.state.job_registry
    entry = registry.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    if entry.status == STATUS_ERROR:
        raise HTTPException(
            status_code=500,
            detail=f"Job failed: {entry.error or 'unknown error'}",
        )
    if entry.status != STATUS_DONE:
        raise HTTPException(
            status_code=425,
            detail=f"Job not done yet (status={entry.status}, stage={entry.stage})",
        )

    settings: Settings = request.app.state.settings
    jobs = JobManager(settings)
    sop_json_path = jobs.sop_json_path(job_id)
    if not sop_json_path.exists():
        raise HTTPException(status_code=500, detail="sop.json missing on disk")
    with open(sop_json_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ═══════════════════════════════════════════════════════════════════════
#  GET /v1/jobs/{job_id}/zip — single-file download bundle
# ═══════════════════════════════════════════════════════════════════════


_STREAM_CHUNK_SIZE = 64 * 1024


@router.get(
    "/jobs/{job_id}/zip",
    responses={
        404: {"model": SOPErrorResponse, "description": "Unknown job_id"},
    },
)
def download_job_zip(job_id: str, request: Request) -> StreamingResponse:
    """Stream the per-job bundle (sop.json + sop.docx + images/) as a zip.

    If a cached ``sop_bundle.zip`` exists in the job dir we stream it as-is;
    otherwise we build it on demand from whatever artifacts are present.
    """
    settings: Settings = request.app.state.settings
    jobs = JobManager(settings)
    job_dir = jobs.job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    zip_path = job_dir / BUNDLE_FILENAME
    if not zip_path.exists():
        write_zip(job_dir, zip_path)

    download_name = _bundle_download_name(jobs.sop_json_path(job_id), job_id)

    def _stream_zip(path: Path):
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk

    return StreamingResponse(
        _stream_zip(zip_path),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"',
        },
    )


def _bundle_download_name(sop_json_path: Path, job_id: str) -> str:
    """Pick a filename for the zip download based on the SOP title."""
    title: Optional[str] = None
    if sop_json_path.exists():
        try:
            with open(sop_json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            title = data.get("sop", {}).get("title")
        except (OSError, json.JSONDecodeError):
            title = None

    slug = slugify(title)
    if slug == "sop":
        return f"sop_{job_id}.zip"
    return f"{slug}.zip"
