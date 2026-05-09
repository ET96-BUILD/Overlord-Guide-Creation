"""Tests for the folder-based batch CLI workflow."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from sopgen.__main__ import cli
from sopgen.api.schemas import SOPDocument


# ── Helpers ─────────────────────────────────────────────────────────────


_RUN_DIR_RE = re.compile(r"__\d{8}-\d{6}$")


def _find_run_dir(out_root: Path, stem: str) -> Path:
    """Return the single timestamped run dir for *stem* under *out_root*."""
    pattern = re.compile(rf"^{re.escape(stem)}__\d{{8}}-\d{{6}}$")
    matches = [
        p for p in out_root.iterdir()
        if p.is_dir() and pattern.fullmatch(p.name)
    ]
    assert len(matches) == 1, (
        f"expected exactly 1 run dir for {stem!r}, found {len(matches)}: "
        f"{[m.name for m in matches]}"
    )
    return matches[0]


def _all_run_dirs(out_root: Path, stem: str) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(stem)}__\d{{8}}-\d{{6}}$")
    return sorted(
        p for p in out_root.iterdir()
        if p.is_dir() and pattern.fullmatch(p.name)
    )


def _fake_sop(title: str = "Test SOP") -> SOPDocument:
    return SOPDocument.model_validate(
        {
            "title": title,
            "intro": "Intro text.",
            "settings": {"max_substeps_per_step": 4, "min_images_per_step": 1},
            "steps": [
                {
                    "step_number": 1,
                    "step_title": "Step one",
                    "substeps": ["Do thing"],
                    "evidence": {
                        "recommended_screenshot_timestamps": ["00:02"],
                        "supporting_timestamps": [],
                    },
                    "images": [],
                }
            ],
            "warnings": [],
        }
    )


def _make_video(path: Path) -> Path:
    """Write a small file with a video extension so MIME detection passes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Bytes are irrelevant — mime.py validates by extension.
    path.write_bytes(b"\x00" * 32)
    return path


@pytest.fixture()
def project_root(tmp_path, monkeypatch):
    """A scratch projects/ root with recording/ pre-created."""
    monkeypatch.setenv("SOPGEN_GEMINI_API_KEY", "test-key")
    root = tmp_path / "projects"
    (root / "recording").mkdir(parents=True)
    return root


@pytest.fixture()
def patched_pipeline():
    """Patch out Gemini + ffmpeg so _process_video runs without external deps.

    By default ``run_with_repair`` returns a fake SOP; per-test overrides can
    set ``side_effect`` to simulate failures.
    """
    with patch("sopgen.__main__.GeminiClient") as gem, \
         patch("sopgen.__main__.VideoAnalyzer") as ana, \
         patch("sopgen.__main__.FFmpegExtractor") as ff, \
         patch("sopgen.__main__.run_with_repair") as rep:
        ff.return_value.extract_all.return_value = {}
        rep.return_value = _fake_sop()
        yield {"gemini": gem, "analyzer": ana, "ffmpeg": ff, "repair": rep}


# ═══════════════════════════════════════════════════════════════════════
#  Happy path — 2 videos, both succeed
# ═══════════════════════════════════════════════════════════════════════


class TestBatchHappyPath:
    def test_two_videos_processed_and_moved(self, project_root, patched_pipeline):
        recording = project_root / "recording"
        _make_video(recording / "alpha.mp4")
        _make_video(recording / "bravo.mp4")

        result = CliRunner().invoke(cli, ["run", "--folder", str(recording)])
        assert result.exit_code == 0, result.output

        # Outputs created in timestamped run dirs
        alpha_run = _find_run_dir(project_root / "out", "alpha")
        bravo_run = _find_run_dir(project_root / "out", "bravo")
        assert (alpha_run / "sop.json").exists()
        assert (bravo_run / "sop.json").exists()

        # Originals moved to completed/
        assert (project_root / "completed" / "alpha.mp4").exists()
        assert (project_root / "completed" / "bravo.mp4").exists()
        assert not (recording / "alpha.mp4").exists()
        assert not (recording / "bravo.mp4").exists()

        # Summary line
        assert "Summary: 2 processed, 0 failed" in result.output


# ═══════════════════════════════════════════════════════════════════════
#  Skip non-video + dotfiles
# ═══════════════════════════════════════════════════════════════════════


class TestBatchSkips:
    def test_dotfiles_and_non_video_skipped(self, project_root, patched_pipeline):
        recording = project_root / "recording"
        _make_video(recording / "good.mp4")
        # Should be ignored
        (recording / ".hidden.mp4").write_bytes(b"\x00")
        (recording / "notes.txt").write_text("hello")
        (recording / "README.md").write_text("# readme")

        result = CliRunner().invoke(cli, ["run", "--folder", str(recording)])
        assert result.exit_code == 0, result.output

        # Only the .mp4 was processed
        good_run = _find_run_dir(project_root / "out", "good")
        assert (good_run / "sop.json").exists()
        assert (project_root / "completed" / "good.mp4").exists()

        # Non-videos stay in recording/
        assert (recording / ".hidden.mp4").exists()
        assert (recording / "notes.txt").exists()
        assert (recording / "README.md").exists()

        assert "Summary: 1 processed, 0 failed" in result.output


# ═══════════════════════════════════════════════════════════════════════
#  Partial failure — one video fails, the other still processes
# ═══════════════════════════════════════════════════════════════════════


class TestBatchPartialFailure:
    def test_failed_video_stays_in_recording(self, project_root, patched_pipeline):
        recording = project_root / "recording"
        _make_video(recording / "good.mp4")
        _make_video(recording / "bad.mp4")

        def selective_fail(analyzer, video_path, mime_type, **kwargs):
            if "bad" in Path(video_path).name:
                raise RuntimeError("simulated Gemini failure")
            return _fake_sop()

        patched_pipeline["repair"].side_effect = selective_fail
        patched_pipeline["repair"].return_value = None  # ignored when side_effect is set

        result = CliRunner().invoke(cli, ["run", "--folder", str(recording)])

        # Non-zero exit code signals at least one failure (exit 2)
        assert result.exit_code == 2, result.output

        # Good video processed and moved
        good_run = _find_run_dir(project_root / "out", "good")
        assert (good_run / "sop.json").exists()
        assert (project_root / "completed" / "good.mp4").exists()
        assert not (recording / "good.mp4").exists()

        # Bad video remains in recording/, no completed copy, no run dir
        assert (recording / "bad.mp4").exists()
        assert not (project_root / "completed" / "bad.mp4").exists()
        assert _all_run_dirs(project_root / "out", "bad") == []

        assert "Summary: 1 processed, 1 failed" in result.output
        assert "bad.mp4" in result.output


# ═══════════════════════════════════════════════════════════════════════
#  Filename collision in completed/ → timestamp suffix
# ═══════════════════════════════════════════════════════════════════════


class TestCompletedCollision:
    def test_collision_gets_timestamp_suffix(self, project_root, patched_pipeline):
        recording = project_root / "recording"
        completed = project_root / "completed"
        completed.mkdir()

        # Pre-existing file with the same name in completed/
        existing = completed / "report.mp4"
        existing.write_bytes(b"OLD")

        _make_video(recording / "report.mp4")

        result = CliRunner().invoke(cli, ["run", "--folder", str(recording)])
        assert result.exit_code == 0, result.output

        # Old file untouched
        assert existing.exists()
        assert existing.read_bytes() == b"OLD"

        # New file moved with timestamp suffix
        suffixed = [
            p for p in completed.iterdir()
            if p.name != "report.mp4" and p.suffix == ".mp4"
        ]
        assert len(suffixed) == 1
        # Format: report_YYYYMMDD-HHMMSS.mp4
        name = suffixed[0].name
        assert name.startswith("report_")
        assert name.endswith(".mp4")
        # 8 digits + dash + 6 digits between
        stamp = name[len("report_"):-len(".mp4")]
        assert len(stamp) == 15 and stamp[8] == "-"
        assert stamp.replace("-", "").isdigit()


# ═══════════════════════════════════════════════════════════════════════
#  Timestamped run directories — re-run produces a fresh dir
# ═══════════════════════════════════════════════════════════════════════


from tests._pngs import write_fake_png as _write_real_png


def _populated_extract_all(video_path, timestamps, output_dir):
    """Mock for FFmpegExtractor.extract_all that writes one valid PNG.

    Returns ``{ts: path}`` so the packager copies it into images/ and
    each run dir ends up with a populated images/ subfolder. The PNG
    is structurally valid because the docx side-output verifies its
    chunk CRCs while embedding.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fake = output_dir / "frame_000.png"
    _write_real_png(fake)
    return {ts: fake for ts in timestamps}


class TestTimestampedRunDirs:
    def test_dir_name_matches_stem_and_timestamp_pattern(
        self, project_root, patched_pipeline
    ):
        recording = project_root / "recording"
        _make_video(recording / "alpha.mp4")

        result = CliRunner().invoke(cli, ["run", "--folder", str(recording)])
        assert result.exit_code == 0, result.output

        run = _find_run_dir(project_root / "out", "alpha")
        assert re.fullmatch(r"alpha__\d{8}-\d{6}", run.name)

    def test_same_video_twice_produces_two_distinct_run_dirs(
        self, project_root, patched_pipeline
    ):
        """Re-uploading and re-running the same video must never overwrite
        the previous run — each run lands in a fresh timestamped directory."""
        recording = project_root / "recording"
        out_root = project_root / "out"

        # Populate images/ in each run so we exercise the full output shape.
        patched_pipeline["ffmpeg"].return_value.extract_all.side_effect = (
            _populated_extract_all
        )

        # Drive the clock so two back-to-back runs land on distinct seconds.
        # 2 calls in _process_video (one per run) + 1 collision call when
        # moving the second video.mp4 over the first in completed/.
        fake_times = [
            datetime(2026, 5, 8, 12, 0, 0),  # run 1: timestamp suffix
            datetime(2026, 5, 8, 12, 0, 1),  # run 2: timestamp suffix
            datetime(2026, 5, 8, 12, 0, 2),  # run 2: collision suffix on move
        ]

        with patch("sopgen.__main__.datetime") as mock_dt:
            mock_dt.now.side_effect = fake_times

            # Run 1
            _make_video(recording / "report.mp4")
            r1 = CliRunner().invoke(cli, ["run", "--folder", str(recording)])
            assert r1.exit_code == 0, r1.output

            # User drops the same-named video again, runs again
            _make_video(recording / "report.mp4")
            r2 = CliRunner().invoke(cli, ["run", "--folder", str(recording)])
            assert r2.exit_code == 0, r2.output

        runs = _all_run_dirs(out_root, "report")
        assert len(runs) == 2, [r.name for r in runs]
        assert runs[0].name == "report__20260508-120000"
        assert runs[1].name == "report__20260508-120001"

        # Both runs have valid sop.json and a populated images/ subfolder
        for r in runs:
            sop_json = r / "sop.json"
            assert sop_json.exists() and sop_json.stat().st_size > 0
            images = r / "images"
            assert images.is_dir()
            pngs = list(images.glob("*.png"))
            assert pngs, f"images/ in {r} should be populated"


# ═══════════════════════════════════════════════════════════════════════
#  Single-file mode: --out is honored exactly (no timestamp surprise)
# ═══════════════════════════════════════════════════════════════════════


class TestSingleFileNoTimestamp:
    def test_single_file_writes_to_explicit_out_dir(
        self, project_root, tmp_path, patched_pipeline
    ):
        video = _make_video(tmp_path / "x.mp4")
        out = tmp_path / "user-named-out"

        result = CliRunner().invoke(
            cli, ["run", "--video", str(video), "--out", str(out)]
        )
        assert result.exit_code == 0, result.output

        # sop.json lands at exactly the user-named path — no timestamp suffix.
        assert (out / "sop.json").exists()
        # And no sibling timestamped dir was created.
        assert not list(out.parent.glob("user-named-out__*"))


# ═══════════════════════════════════════════════════════════════════════
#  Mutually exclusive flags
# ═══════════════════════════════════════════════════════════════════════


class TestFlagValidation:
    def test_video_and_folder_rejected(self, project_root, tmp_path, patched_pipeline):
        recording = project_root / "recording"
        video = _make_video(tmp_path / "x.mp4")

        result = CliRunner().invoke(
            cli,
            ["run", "--video", str(video), "--folder", str(recording)],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output.lower()

    def test_video_without_out_rejected(self, project_root, tmp_path, patched_pipeline):
        video = _make_video(tmp_path / "x.mp4")
        result = CliRunner().invoke(cli, ["run", "--video", str(video)])
        assert result.exit_code != 0
        assert "--out is required" in result.output
