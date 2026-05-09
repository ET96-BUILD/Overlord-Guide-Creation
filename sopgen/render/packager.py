"""Merge the validated SOP document with extracted screenshot images."""

from __future__ import annotations

import base64
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from sopgen.api.schemas import SOPDocument

logger = logging.getLogger(__name__)


class SOPPackager:
    """Combines SOP JSON + frame files into the final deliverable."""

    # ── Public API ──────────────────────────────────────────────────────

    def collect_timestamps(self, sop: SOPDocument) -> list[str]:
        """Return a flat, deduplicated list of all screenshot timestamps."""
        seen: set[str] = set()
        ordered: list[str] = []
        for step in sop.steps:
            for ts in step.evidence.recommended_screenshot_timestamps:
                if ts not in seen:
                    seen.add(ts)
                    ordered.append(ts)
        return ordered

    def package(
        self,
        sop: SOPDocument,
        frame_map: dict[str, Path],
        job_id: str,
        output_images_dir: Path,
        *,
        embed_base64: bool = False,
        image_base_url: Optional[str] = None,
    ) -> dict:
        """Build the API response dict.

        Parameters
        ----------
        sop : SOPDocument
            Validated SOP.
        frame_map : dict[str, Path]
            ``{timestamp: local_path}`` from ffmpeg extraction. The raw
            timestamped frames are kept on disk as debug artifacts; this
            method *copies* each one to ``<image_id>.png`` in
            *output_images_dir* so the user-facing filenames match the
            ``image_id`` field in the response.
        job_id : str
            Job identifier.
        output_images_dir : Path
            Directory where the renamed ``<image_id>.png`` copies will be
            written. Created if it does not exist.
        embed_base64 : bool
            If ``True``, embed image bytes as ``data:image/png;base64,...``
            strings in each image entry (useful for CLI / offline output).
        image_base_url : str, optional
            Override the URL prefix used in the response. Defaults to
            ``/static/jobs/<job_id>/images`` (the API static mount). CLI
            callers typically pass ``"images"`` for paths relative to the
            generated ``sop.json``.
        """
        sop_dict = sop.model_dump()
        if image_base_url is None:
            image_base_url = f"/static/jobs/{job_id}/images"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        all_images: list[dict] = []

        for step in sop_dict["steps"]:
            step_num = step["step_number"]
            ts_list = step["evidence"]["recommended_screenshot_timestamps"]

            # Build / update images list for this step
            populated_images: list[dict] = []
            for img_idx, ts in enumerate(ts_list, start=1):
                image_id = f"step_{step_num}_img_{img_idx}"
                caption = _find_caption(step["images"], image_id, step["step_title"], ts)
                filename = f"{image_id}.png"

                entry: dict = {"image_id": image_id, "caption": caption}
                src = frame_map.get(ts)
                if src is not None and src.exists():
                    dest = output_images_dir / filename
                    shutil.copyfile(src, dest)
                    entry["filename"] = filename
                    entry["url"] = f"{image_base_url}/{filename}"
                    if embed_base64:
                        entry["base64_data"] = _encode(dest)
                else:
                    entry["filename"] = ""
                    entry["url"] = ""
                    logger.warning(
                        "No frame for step %d ts=%s", step_num, ts
                    )

                populated_images.append(entry)
                all_images.append(entry)

            step["images"] = populated_images

        return {
            "job_id": job_id,
            "sop": sop_dict,
            "image_base_url": image_base_url,
            "images": all_images,
        }

    def save(self, package: dict, path: Path) -> None:
        """Write the packaged SOP to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(package, fh, indent=2, ensure_ascii=False)
        logger.info("Saved SOP package → %s", path)


# ── Helpers ─────────────────────────────────────────────────────────────


def _find_caption(
    images: list[dict], image_id: str, step_title: str, ts: str
) -> str:
    """Return the Gemini-supplied caption for *image_id*, or a fallback."""
    for img in images:
        if img.get("image_id") == image_id and img.get("caption"):
            return img["caption"]
    return f"Screenshot at {ts} — {step_title}"


def _encode(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"
