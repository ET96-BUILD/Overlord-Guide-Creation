"""FFmpeg-based frame extraction at specific timestamps.

Also exposes ``probe_codec`` and ``transcode_to_h264`` so the pipeline
can normalise inputs Gemini's Files API can't decode (HEVC out of the
Windows Snipping Tool, ProRes from Final Cut, etc.) into H.264 before
upload.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_EXTRACT_TIMEOUT_S = 30
_PROBE_TIMEOUT_S = 30
_TRANSCODE_TIMEOUT_S = 900  # 15 min ceiling for very long recordings

# Codecs Gemini's Files API decodes reliably. Anything outside this set
# gets normalised to H.264 before upload. HEVC is officially listed by
# Google but in practice the Snipping Tool / Loom / Zoom variants of it
# regularly fail server-side processing, so we transcode it too.
_GEMINI_FRIENDLY_CODECS = frozenset({"h264", "vp8", "vp9", "av1"})


def is_gemini_friendly_codec(codec: Optional[str]) -> bool:
    """True if *codec* (as ffprobe reports it) doesn't need transcoding."""
    if not codec:
        # Unknown codec → safer to leave alone; Gemini may still accept it.
        return True
    return codec.lower() in _GEMINI_FRIENDLY_CODECS


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

    # ── Codec probe + transcode (pre-upload normalisation) ─────────────

    def probe_codec(self, video_path: Path) -> Optional[str]:
        """Return the video stream's codec name, or None if probe fails.

        Uses ``ffprobe`` (assumed to live next to ``ffmpeg`` on PATH —
        the apt-get ffmpeg package ships both binaries).

        All failure modes return None and log a WARNING with the reason.
        Downstream code treats None as Gemini-friendly (fail-open: skip
        the transcode pass), so without the log a real ffprobe failure
        would silently let an HEVC file slip through to Gemini and 502
        with no breadcrumb. The warnings make that path diagnosable in
        Cloud Run logs.
        """
        ffprobe = self._ffprobe_path()
        cmd = [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning(
                "Could not probe codec for %s (%s); treating as "
                "Gemini-friendly and skipping transcode",
                video_path, exc,
            )
            return None
        if result.returncode != 0:
            logger.warning(
                "Could not probe codec for %s (ffprobe rc=%d: %s); "
                "treating as Gemini-friendly and skipping transcode",
                video_path, result.returncode, result.stderr[:200].strip(),
            )
            return None
        codec = result.stdout.strip()
        if not codec:
            logger.warning(
                "Could not probe codec for %s (ffprobe reported no codec "
                "name); treating as Gemini-friendly and skipping transcode",
                video_path,
            )
            return None
        return codec

    def transcode_to_h264(
        self, input_path: Path, output_path: Path
    ) -> bool:
        """Re-encode *input_path* to H.264 + AAC mp4 at *output_path*.

        Returns True on success. Picks fast-but-decent defaults: CRF 23,
        ``faststart`` so the moov atom is at the head of the file (Gemini
        starts processing earlier), AAC audio.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(output_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_TRANSCODE_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error("ffmpeg transcode failed to spawn: %s", exc)
            return False
        if result.returncode != 0:
            logger.error(
                "ffmpeg transcode rc=%d: %s",
                result.returncode,
                result.stderr[-500:],
            )
            return False
        return output_path.exists() and output_path.stat().st_size > 0

    def _ffprobe_path(self) -> str:
        """Derive the ffprobe binary path from ffmpeg_path.

        Most distributions ship ffprobe alongside ffmpeg in the same
        directory. If the user passed a bare ``ffmpeg`` (default), we
        return ``ffprobe`` and let PATH resolve it; if they passed an
        absolute path like ``/usr/local/bin/ffmpeg``, swap the basename.
        """
        path = Path(self.ffmpeg_path)
        if path.name in ("ffmpeg", "ffmpeg.exe"):
            sibling = path.with_name("ffprobe" + (".exe" if path.suffix == ".exe" else ""))
            return str(sibling) if path.parent != Path("") else "ffprobe"
        return "ffprobe"

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
