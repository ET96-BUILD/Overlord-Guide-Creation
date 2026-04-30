"""FFmpeg-based frame extraction at specific timestamps."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_EXTRACT_TIMEOUT_S = 30


class FFmpegExtractor:
    """Extracts individual frames from a video using ffmpeg."""

    def __init__(self, ffmpeg_path: str = "ffmpeg") -> None:
        self.ffmpeg_path = ffmpeg_path

    # ── Single frame ────────────────────────────────────────────────────

    def extract_frame(
        self,
        video_path: Path,
        timestamp: str,
        output_path: Path,
    ) -> bool:
        """Extract one frame at *timestamp* (``MM:SS`` or ``HH:MM:SS``).

        Returns ``True`` on success, ``False`` on any failure.
        """
        cmd = [
            self.ffmpeg_path,
            "-ss", timestamp,
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            "-y",
            str(output_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_EXTRACT_TIMEOUT_S,
            )
            if result.returncode != 0:
                logger.warning(
                    "ffmpeg failed (rc=%d) for ts=%s: %s",
                    result.returncode,
                    timestamp,
                    result.stderr[:200],
                )
                return False
            return output_path.exists() and output_path.stat().st_size > 0
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out for ts=%s", timestamp)
            return False
        except FileNotFoundError:
            logger.error("ffmpeg not found at %s", self.ffmpeg_path)
            return False

    # ── Batch extraction ────────────────────────────────────────────────

    def extract_all(
        self,
        video_path: Path,
        timestamps: list[str],
        output_dir: Path,
    ) -> dict[str, Path]:
        """Extract frames for every timestamp.

        Returns a mapping ``{timestamp: output_path}`` for successful
        extractions only.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted: dict[str, Path] = {}

        for i, ts in enumerate(timestamps):
            safe = ts.replace(":", "_")
            out = output_dir / f"frame_{i:03d}_{safe}.png"
            if self.extract_frame(video_path, ts, out):
                extracted[ts] = out
            else:
                logger.warning("Skipped frame for ts=%s", ts)

        logger.info(
            "Extracted %d/%d frames", len(extracted), len(timestamps)
        )
        return extracted

    # ── Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def parse_timestamp(ts: str) -> int:
        """Convert ``MM:SS`` or ``HH:MM:SS`` to total seconds."""
        parts = ts.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        raise ValueError(f"Invalid timestamp format: {ts}")
