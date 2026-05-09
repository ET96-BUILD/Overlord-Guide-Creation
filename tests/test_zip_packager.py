"""Tests for the zip bundle output."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from sopgen.render.zip_packager import BUNDLE_FILENAME, slugify, write_zip


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def populated_output(tmp_path: Path) -> Path:
    """An output_dir with the artifacts a real run would produce."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "sop.json").write_text('{"sop": {"title": "Test SOP"}}', encoding="utf-8")
    (out / "sop.docx").write_bytes(b"PK\x03\x04...fake docx bytes")
    images = out / "images"
    images.mkdir()
    (images / "step_1_img_1.png").write_bytes(b"\x89PNG fake-1")
    (images / "step_2_img_1.png").write_bytes(b"\x89PNG fake-2")
    return out


# ── write_zip ───────────────────────────────────────────────────────────


class TestWriteZip:
    def test_zip_contains_expected_members(self, populated_output, tmp_path):
        zip_path = populated_output / BUNDLE_FILENAME
        result = write_zip(populated_output, zip_path)
        assert result == zip_path
        assert zip_path.exists() and zip_path.stat().st_size > 0

        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())

        assert "sop.json" in names
        assert "sop.docx" in names
        assert "images/step_1_img_1.png" in names
        assert "images/step_2_img_1.png" in names

    def test_arcnames_use_forward_slashes(self, populated_output):
        """Portable zips use POSIX separators regardless of platform."""
        zip_path = populated_output / BUNDLE_FILENAME
        write_zip(populated_output, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                assert "\\" not in name, f"backslash in arcname: {name!r}"

    def test_zip_can_be_written_outside_output_dir(self, populated_output, tmp_path):
        zip_path = tmp_path / "elsewhere" / "bundle.zip"
        write_zip(populated_output, zip_path)
        assert zip_path.exists()
        with zipfile.ZipFile(zip_path) as zf:
            assert "sop.json" in zf.namelist()

    def test_missing_output_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            write_zip(tmp_path / "does-not-exist", tmp_path / "out.zip")


# ── Recursion guard ─────────────────────────────────────────────────────


class TestRecursionGuard:
    def test_existing_sop_bundle_is_excluded_from_new_zip(
        self, populated_output
    ):
        """A stale sop_bundle.zip from a previous run must never be included
        inside a fresh sop_bundle.zip."""
        zip_path = populated_output / BUNDLE_FILENAME

        # Pre-place a fake old bundle in the output dir.
        zip_path.write_bytes(b"OLD-BUNDLE-CONTENT")

        write_zip(populated_output, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())

        # The new zip overwrote the old one and does NOT contain itself
        # as a member.
        assert BUNDLE_FILENAME not in names
        assert "sop.json" in names

    def test_no_self_inclusion_when_writing_inside_output_dir(
        self, populated_output
    ):
        """Even with no pre-existing bundle, writing the zip into the
        output_dir must not pick up the in-progress file as a member."""
        zip_path = populated_output / BUNDLE_FILENAME
        write_zip(populated_output, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            assert BUNDLE_FILENAME not in zf.namelist()


# ── slugify ─────────────────────────────────────────────────────────────


class TestSlugify:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("How to Pay EAL Commission", "how-to-pay-eal-commission"),
            ("Already-slug", "already-slug"),
            ("  trim me  ", "trim-me"),
            ("multi   spaces!!!", "multi-spaces"),
            ("Mixed/Slashes_and.dots", "mixed-slashes-and-dots"),
            ("UPPER", "upper"),
            ("---leading---", "leading"),
        ],
    )
    def test_basic_cases(self, raw, expected):
        assert slugify(raw) == expected

    def test_empty_returns_sop_fallback(self):
        assert slugify("") == "sop"
        assert slugify(None) == "sop"
        assert slugify("    ") == "sop"
        assert slugify("!!!") == "sop"

    def test_caps_at_max_len_and_strips_trailing_hyphen(self):
        # 80 chars of "a-" pattern → cap at 60 then rstrip("-")
        long_input = "a-" * 80
        out = slugify(long_input, max_len=60)
        assert len(out) <= 60
        assert not out.endswith("-")
