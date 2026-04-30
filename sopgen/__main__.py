"""CLI entry point — ``python -m sopgen run --video … --out …``."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from sopgen.core.config import Settings
from sopgen.core.ffmpeg import FFmpegExtractor
from sopgen.core.mime import validate_video_file, supported_types_list
from sopgen.core.validation import run_with_repair
from sopgen.gemini.client import GeminiClient
from sopgen.gemini.video_analyze import VideoAnalyzer
from sopgen.render.packager import SOPPackager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


@click.group()
def cli() -> None:
    """SOP Generator — turn screen recordings into structured SOPs."""


@cli.command()
@click.option("--video", required=True, type=click.Path(exists=True), help="Path to video file.")
@click.option("--out", required=True, type=click.Path(), help="Output directory.")
@click.option("--title-hint", default=None, help="Suggested SOP title.")
@click.option("--domain-hint", default=None, help='Domain context, e.g. "NetSuite AP process".')
@click.option("--media-resolution", default=None, type=click.Choice(["low", "default"]))
@click.option("--fps-override", default=None, type=int, help="Override 1 FPS default.")
def run(
    video: str,
    out: str,
    title_hint: str | None,
    domain_hint: str | None,
    media_resolution: str | None,
    fps_override: int | None,
) -> None:
    """Generate an SOP from a screen recording video."""

    video_path = Path(video)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings()
    if media_resolution:
        settings = settings.model_copy(update={"gemini_media_resolution": media_resolution})
    if fps_override is not None:
        settings = settings.model_copy(update={"gemini_video_fps_override": fps_override})

    # ── Validate ────────────────────────────────────────────────────
    click.echo(f"Video: {video_path}")
    is_valid, mime_type = validate_video_file(video_path)
    if not is_valid:
        click.echo(
            f"Error: unsupported format ({mime_type}). "
            f"Supported: {supported_types_list()}",
            err=True,
        )
        sys.exit(1)
    click.echo(f"MIME type: {mime_type}")

    # ── Analyze ─────────────────────────────────────────────────────
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
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"SOP validated — {len(sop.steps)} steps")

    # ── Extract frames ──────────────────────────────────────────────
    click.echo("Extracting screenshots…")
    packager = SOPPackager()
    timestamps = packager.collect_timestamps(sop)

    images_dir = out_dir / "images"
    ffmpeg = FFmpegExtractor(settings.ffmpeg_path)
    frame_map = ffmpeg.extract_all(video_path, timestamps, images_dir)
    click.echo(f"Extracted {len(frame_map)}/{len(timestamps)} frames")

    # ── Package ─────────────────────────────────────────────────────
    package = packager.package(sop, frame_map, "cli", embed_base64=True)
    sop_path = out_dir / "sop.json"
    with open(sop_path, "w", encoding="utf-8") as fh:
        json.dump(package, fh, indent=2, ensure_ascii=False)

    click.echo(f"SOP saved → {sop_path}")


if __name__ == "__main__":
    cli()
