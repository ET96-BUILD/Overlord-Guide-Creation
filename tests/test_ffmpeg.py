"""Tests for ffmpeg utilities (timestamp parsing + frame extraction)."""

import shutil
import subprocess

import pytest

from sopgen.core.ffmpeg import FFmpegExtractor, is_gemini_friendly_codec


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


# ── Codec probe + transcode (pre-upload normalisation) ──────────────────


class TestIsGeminiFriendlyCodec:
    @pytest.mark.parametrize(
        "codec, expected",
        [
            ("h264", True),
            ("H264", True),       # case-insensitive
            ("vp8", True),
            ("vp9", True),
            ("av1", True),
            ("hevc", False),      # Snipping Tool / iPhone output
            ("h265", False),
            ("prores", False),    # Final Cut output
            ("mpeg4", False),
        ],
    )
    def test_known_codecs(self, codec, expected):
        assert is_gemini_friendly_codec(codec) is expected

    def test_unknown_codec_defaults_to_friendly(self):
        """If ffprobe can't tell us what the codec is, leave it alone and
        let Gemini decide — we'd rather fail at Gemini than waste a
        transcode pass on an unknown."""
        assert is_gemini_friendly_codec(None) is True
        assert is_gemini_friendly_codec("") is True


@pytest.mark.skipif(not _ffmpeg_available, reason="ffmpeg not installed")
class TestProbeCodec:
    def test_probes_h264_test_video(self, test_video):
        codec = FFmpegExtractor().probe_codec(test_video)
        # The test fixture writes an mp4 (libx264 default). Codec should
        # be h264 — though some ffmpeg builds report it as 'h264' literal.
        assert codec is not None
        assert codec.lower() == "h264"

    def test_missing_file_returns_none(self, tmp_path):
        codec = FFmpegExtractor().probe_codec(tmp_path / "does-not-exist.mp4")
        assert codec is None


@pytest.mark.skipif(not _ffmpeg_available, reason="ffmpeg not installed")
class TestTranscodeToH264:
    def test_transcodes_test_video(self, test_video, tmp_path):
        """The fixture is already h264, but transcoding re-encodes it and
        verifies the round-trip succeeds + the output is also h264."""
        extractor = FFmpegExtractor()
        out = tmp_path / "transcoded.mp4"
        assert extractor.transcode_to_h264(test_video, out)
        assert out.exists() and out.stat().st_size > 0
        # Output should also be h264
        assert extractor.probe_codec(out).lower() == "h264"

    def test_transcode_failure_returns_false(self, tmp_path):
        """Pointing at a non-existent input must not raise; it just
        returns False so the pipeline can fall back to the original."""
        extractor = FFmpegExtractor()
        out = tmp_path / "nope.mp4"
        ok = extractor.transcode_to_h264(tmp_path / "missing.mp4", out)
        assert ok is False
        assert not out.exists()
