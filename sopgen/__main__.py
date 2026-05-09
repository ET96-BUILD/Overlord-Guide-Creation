"""CLI entry point — single-file (``--video``) and batch (``--folder``) modes."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from sopgen.core.config import Settings
from sopgen.core.ffmpeg import FFmpegExtractor
from sopgen.core.mime import validate_video_file, supported_types_list
from sopgen.core.validation import run_with_repair
from sopgen.gemini.client import GeminiClient
from sopgen.gemini.video_analyze import VideoAnalyzer
from sopgen.render.docx_packager import write_docx
from sopgen.render.packager import SOPPackager
from sopgen.render.zip_packager import write_zip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Default project layout ──────────────────────────────────────────────
DEFAULT_PROJECT_ROOT = Path("projects")
DEFAULT_RECORDING_DIR = DEFAULT_PROJECT_ROOT / "recording"
RECORDING_SUBDIR = "recording"
COMPLETED_SUBDIR = "completed"
OUT_SUBDIR = "out"


@click.group()
def cli() -> None:
    """SOP Generator — turn screen recordings into structured SOPs."""


@cli.command()
@click.option(
    "--video",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a single video file. Mutually exclusive with --folder.",
)
@click.option(
    "--folder",
    default=None,
    type=click.Path(file_okay=False),
    help=(
        "Folder containing videos to process in batch. Defaults to "
        f"{DEFAULT_RECORDING_DIR.as_posix()} when neither --video nor --folder is given. "
        "Outputs go to <folder-parent>/out/<video-stem>/ and successfully "
        "processed videos are moved to <folder-parent>/completed/."
    ),
)
@click.option(
    "--out",
    default=None,
    type=click.Path(),
    help="Output directory (single-file mode only; ignored in batch mode).",
)
@click.option("--title-hint", default=None, help="Suggested SOP title.")
@click.option("--domain-hint", default=None, help='Domain context, e.g. "NetSuite AP process".')
@click.option("--media-resolution", default=None, type=click.Choice(["low", "default"]))
@click.option("--fps-override", default=None, type=int, help="Override 1 FPS default.")
def run(
    video: Optional[str],
    folder: Optional[str],
    out: Optional[str],
    title_hint: Optional[str],
    domain_hint: Optional[str],
    media_resolution: Optional[str],
    fps_override: Optional[int],
) -> None:
    """Generate an SOP from a screen recording.

    Two modes:

    \b
      Single-file:  python -m sopgen run --video file.mp4 --out ./out
      Batch (folder): python -m sopgen run [--folder PATH]
                      processes every video in the folder; outputs land in
                      <folder-parent>/out/<stem>/ and processed videos move
                      to <folder-parent>/completed/.
    """
    if video and folder:
        raise click.UsageError("--video and --folder are mutually exclusive.")

    settings = _build_settings(media_resolution, fps_override)

    if video:
        if not out:
            raise click.UsageError("--out is required when --video is provided.")
        try:
            # Single-file mode honors --out exactly — no timestamp surprise.
            _process_video(
                Path(video),
                Path(out),
                settings=settings,
                title_hint=title_hint,
                domain_hint=domain_hint,
                fps_override=fps_override,
                timestamp_suffix=False,
            )
        except _PipelineError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        return

    # Batch mode
    if out is not None:
        click.echo(
            "Warning: --out is ignored in batch mode; outputs go to "
            "<folder-parent>/out/<video-stem>/.",
            err=True,
        )
    folder_path = Path(folder) if folder else DEFAULT_RECORDING_DIR
    failures = _run_batch(
        folder_path,
        settings=settings,
        title_hint=title_hint,
        domain_hint=domain_hint,
        fps_override=fps_override,
    )
    if failures:
        sys.exit(2)


# ═══════════════════════════════════════════════════════════════════════
#  Per-video pipeline
# ═══════════════════════════════════════════════════════════════════════


class _PipelineError(RuntimeError):
    """Raised by _process_video when a single video cannot be processed."""


def _process_video(
    video_path: Path,
    out_dir: Path,
    *,
    settings: Settings,
    title_hint: Optional[str] = None,
    domain_hint: Optional[str] = None,
    fps_override: Optional[int] = None,
    timestamp_suffix: bool = False,
) -> Path:
    """Run the full pipeline for one video. Returns the sop.json path.

    When *timestamp_suffix* is True, *out_dir* is rewritten to
    ``<out_dir>__<YYYYMMDD-HHMMSS>`` so each run lands in a fresh
    directory — used by batch mode to defeat upload-side caching of
    previously generated sop.json files.

    Raises ``_PipelineError`` on unsupported MIME, validation exhaustion,
    or any underlying Gemini / IO failure.
    """
    # Capture the run timestamp once at the start so any path derived
    # within this call shares the same value.
    run_started_at = datetime.now()
    if timestamp_suffix:
        ts = run_started_at.strftime("%Y%m%d-%H%M%S")
        out_dir = out_dir.with_name(f"{out_dir.name}__{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        click.echo(f"Video: {video_path}")
        is_valid, mime_type = validate_video_file(video_path)
        if not is_valid:
            raise _PipelineError(
                f"unsupported format ({mime_type}). "
                f"Supported: {supported_types_list()}"
            )
        click.echo(f"MIME type: {mime_type}")

        click.echo("Analyzing video with Gemini…")
        gemini = GeminiClient(settings)
        analyzer = VideoAnalyzer(gemini, settings)

        try:
            sop = run_with_repair(
                analyzer,
                video_path,
                mime_type,
                title_hint=title_hint,
                domain_hint=domain_hint,
                fps_override=fps_override,
                max_retries=settings.max_retry_attempts,
            )
        except ValueError as exc:
            raise _PipelineError(str(exc)) from exc

        click.echo(f"SOP validated — {len(sop.steps)} steps")

        click.echo("Extracting screenshots…")
        packager = SOPPackager()
        timestamps = packager.collect_timestamps(sop)

        # Raw timestamped frames go to a hidden debug dir so the user-facing
        # ./images/ folder contains only <image_id>.png files.
        raw_frames_dir = out_dir / ".frames"
        images_dir = out_dir / "images"
        ffmpeg = FFmpegExtractor(settings.ffmpeg_path)
        frame_map = ffmpeg.extract_all(video_path, timestamps, raw_frames_dir)
        click.echo(f"Extracted {len(frame_map)}/{len(timestamps)} frames")

        package = packager.package(
            sop,
            frame_map,
            "cli",
            output_images_dir=images_dir,
            embed_base64=True,
            image_base_url="images",
        )
        sop_path = out_dir / "sop.json"
        with open(sop_path, "w", encoding="utf-8") as fh:
            json.dump(package, fh, indent=2, ensure_ascii=False)

        # Side-output for non-technical users: a text-only Word doc next
        # to sop.json they can open and copy-paste from into Overlord /
        # the iFixit form. Screenshots live alongside in images/.
        docx_path = out_dir / "sop.docx"
        write_docx(package["sop"], docx_path)

        # One-click delivery: bundle sop.json + sop.docx + images/ into a
        # single zip. Built last so it can include the docx that was
        # written the line above.
        zip_path = out_dir / "sop_bundle.zip"
        write_zip(out_dir, zip_path)
    except Exception:
        # Don't leave empty/partial timestamped dirs behind in batch mode —
        # they'd accumulate as graveyard clutter under projects/out/.
        # Single-file mode (timestamp_suffix=False) preserves the user's
        # explicit --out dir on failure.
        if timestamp_suffix and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        raise

    # Print resolved paths so users can copy-paste them reliably into
    # external tools (Cowork, chat) without ambiguity about the cwd.
    click.echo(f"SOP saved -> {sop_path.resolve()}")
    click.echo(f"DOCX saved -> {docx_path.resolve()}")
    click.echo(f"ZIP saved -> {zip_path.resolve()}")
    return sop_path


def _build_settings(
    media_resolution: Optional[str], fps_override: Optional[int]
) -> Settings:
    settings = Settings()
    if media_resolution:
        settings = settings.model_copy(update={"gemini_media_resolution": media_resolution})
    if fps_override is not None:
        settings = settings.model_copy(update={"gemini_video_fps_override": fps_override})
    return settings


# ═══════════════════════════════════════════════════════════════════════
#  Batch mode
# ═══════════════════════════════════════════════════════════════════════


def _run_batch(
    folder: Path,
    *,
    settings: Settings,
    title_hint: Optional[str] = None,
    domain_hint: Optional[str] = None,
    fps_override: Optional[int] = None,
) -> list[tuple[str, str]]:
    """Process every supported video in *folder*. Returns the failure list."""
    project_root = folder.parent
    completed_dir = project_root / COMPLETED_SUBDIR
    out_root = project_root / OUT_SUBDIR

    folder.mkdir(parents=True, exist_ok=True)
    completed_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    accepted: list[Path] = []
    for p in candidates:
        ok, _ = validate_video_file(p)
        if ok:
            accepted.append(p)
        else:
            logger.info("[batch] Skipping non-video file: %s", p.name)

    if not accepted:
        click.echo(f"No video files found in {folder}.")
        return []

    click.echo(f"Batch: {len(accepted)} video(s) in {folder}")

    processed: list[str] = []
    failed: list[tuple[str, str]] = []

    for video in accepted:
        out_dir = out_root / video.stem
        click.echo(f"\n── {video.name} ──")
        try:
            _process_video(
                video,
                out_dir,
                settings=settings,
                title_hint=title_hint,
                domain_hint=domain_hint,
                fps_override=fps_override,
                timestamp_suffix=True,
            )
        except Exception as exc:
            logger.exception("[batch] Failure on %s", video.name)
            click.echo(f"  ✗ {video.name}: {exc}", err=True)
            failed.append((video.name, str(exc)))
            continue

        try:
            dest = _move_to_completed(video, completed_dir)
            click.echo(f"  -> moved to {dest}")
            processed.append(video.name)
        except Exception as exc:
            logger.exception("[batch] Move failed for %s", video.name)
            click.echo(f"  ✗ move failed for {video.name}: {exc}", err=True)
            failed.append((video.name, f"move failed: {exc}"))

    # Summary
    click.echo("")
    click.echo(f"Summary: {len(processed)} processed, {len(failed)} failed")
    if processed:
        click.echo("  Processed:")
        for name in processed:
            click.echo(f"    - {name}")
    if failed:
        click.echo("  Failed:")
        for name, reason in failed:
            click.echo(f"    - {name}: {reason}")

    return failed


def _move_to_completed(video: Path, completed_dir: Path) -> Path:
    """Move *video* into *completed_dir*. Adds a timestamp suffix on collision."""
    target = completed_dir / video.name
    if target.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = completed_dir / f"{video.stem}_{ts}{video.suffix}"
    shutil.move(str(video), str(target))
    return target


if __name__ == "__main__":
    cli()
