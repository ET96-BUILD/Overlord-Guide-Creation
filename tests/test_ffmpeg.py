"""Tests for ffmpeg utilities (timestamp parsing + frame extraction)."""

import shutil
import subprocess

import pytest

from sopgen.core.ffmpeg import FFmpegExtractor


# ── Timestamp parsing ───────────────────────────────────────────────────


class TestParseTimestamp:
    def test_mm_ss(self):
        assert FFmpegExtractor.parse_timestamp("01:30") == 90

    def test_hh_mm_ss(self):
        assert FFmpegExtractor.parse_timestamp("01:05:30") == 3930

    def test_zero(self):
        assert FFmpegExtractor.parse_timestamp("00:00") == 0

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid timestamp"):
            FFmpegExtractor.parse_timestamp("123")


# ── Frame extraction (requires ffmpeg + a generated test video) ─────────

_ffmpeg_available = shutil.which("ffmpeg") is not None


@pytest.fixture()
def test_video(tmp_path):
    """Generate a 5-second colour-bars test video via ffmpeg."""
    if not _ffmpeg_available:
        pytest.skip("ffmpeg not installed")
    video = tmp_path / "test.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=5:r=1",
                "-pix_fmt", "yuv420p",
                str(video),
            ],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"Could not invoke ffmpeg: {exc}")
    if not video.exists():
        pytest.skip("Could not generate test video")
    return video


@pytest.mark.skipif(not _ffmpeg_available, reason="ffmpeg not installed")
class TestExtractFrame:
    def test_single_frame(self, test_video, tmp_path):
        extractor = FFmpegExtractor()
        out = tmp_path / "frame.png"
        assert extractor.extract_frame(test_video, "00:02", out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_batch(self, test_video, tmp_path):
        extractor = FFmpegExtractor()
        frames = extractor.extract_all(
            test_video, ["00:01", "00:03"], tmp_path / "frames"
        )
        assert len(frames) == 2
        for path in frames.values():
            assert path.exists()

    def test_out_of_range_timestamp(self, test_video, tmp_path):
        """Timestamp beyond video length — ffmpeg may still produce a frame
        (last frame) or fail; either way we don't crash."""
        extractor = FFmpegExtractor()
        out = tmp_path / "late.png"
        # Just verify no exception is raised
        extractor.extract_frame(test_video, "99:59", out)
