"""Tests for SOPPackager — particularly the image_id-matching filenames."""

from __future__ import annotations

from pathlib import Path

from sopgen.api.schemas import SOPDocument
from sopgen.render.packager import SOPPackager


def _make_sop(num_steps: int = 2, ts_per_step: int = 2) -> SOPDocument:
    steps = []
    ts_counter = 1
    for i in range(1, num_steps + 1):
        timestamps = [f"00:{ts_counter + j:02d}" for j in range(ts_per_step)]
        ts_counter += ts_per_step
        steps.append(
            {
                "step_number": i,
                "step_title": f"Step {i}",
                "substeps": [f"Do thing {i}"],
                "evidence": {
                    "recommended_screenshot_timestamps": timestamps,
                    "supporting_timestamps": [],
                },
                "images": [],
            }
        )
    return SOPDocument.model_validate(
        {
            "title": "Test SOP",
            "intro": "Intro text.",
            "settings": {"max_substeps_per_step": 4, "min_images_per_step": 1},
            "steps": steps,
            "warnings": [],
        }
    )


def _write_fake_frame(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal-but-real PNG header so file size > 0 and shutil.copyfile is happy.
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)


# ── Filename = image_id ─────────────────────────────────────────────────


class TestImageIdFilenames:
    def test_each_image_filename_matches_image_id(self, tmp_path):
        sop = _make_sop(num_steps=2, ts_per_step=2)

        # Build a frame_map covering every timestamp the SOP recommends.
        raw_dir = tmp_path / ".frames"
        frame_map: dict[str, Path] = {}
        for step in sop.steps:
            for ts in step.evidence.recommended_screenshot_timestamps:
                src = raw_dir / f"frame_{ts.replace(':', '_')}.png"
                _write_fake_frame(src)
                frame_map[ts] = src

        out_images = tmp_path / "out" / "images"
        packager = SOPPackager()
        package = packager.package(
            sop,
            frame_map,
            job_id="test_job",
            output_images_dir=out_images,
            image_base_url="images",
        )

        # Every image_id in the response has a file at the reported path,
        # and that file's basename equals <image_id>.png.
        assert package["images"], "expected at least one image entry"
        for img in package["images"]:
            image_id = img["image_id"]
            expected = f"{image_id}.png"
            assert img["filename"] == expected
            assert img["url"] == f"images/{expected}"
            on_disk = out_images / expected
            assert on_disk.exists(), f"missing {on_disk}"
            assert on_disk.name == expected

        # Per-step images list mirrors the response.
        for step in package["sop"]["steps"]:
            for img in step["images"]:
                assert img["filename"] == f"{img['image_id']}.png"

    def test_default_image_base_url_uses_static_jobs_path(self, tmp_path):
        sop = _make_sop(num_steps=1, ts_per_step=1)
        ts = sop.steps[0].evidence.recommended_screenshot_timestamps[0]
        src = tmp_path / ".frames" / "raw.png"
        _write_fake_frame(src)

        package = SOPPackager().package(
            sop,
            {ts: src},
            job_id="abc123",
            output_images_dir=tmp_path / "images",
        )

        assert package["image_base_url"] == "/static/jobs/abc123/images"
        assert package["images"][0]["url"] == "/static/jobs/abc123/images/step_1_img_1.png"

    def test_missing_frame_yields_empty_url_no_copy(self, tmp_path):
        sop = _make_sop(num_steps=1, ts_per_step=1)

        out_images = tmp_path / "images"
        package = SOPPackager().package(
            sop,
            {},  # no frames extracted
            job_id="job",
            output_images_dir=out_images,
        )

        img = package["images"][0]
        assert img["url"] == ""
        assert img["filename"] == ""
        # No file should have been created for this image_id.
        assert not (out_images / f"{img['image_id']}.png").exists()
