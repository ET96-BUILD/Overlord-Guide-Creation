"""Merge the validated SOP document with extracted screenshot images."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

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
        *,
        embed_base64: bool = False,
    ) -> dict:
        """Build the API response dict.

        Parameters
        ----------
        sop : SOPDocument
            Validated SOP.
        frame_map : dict[str, Path]
            ``{timestamp: local_path}`` from ffmpeg extraction.
        job_id : str
            Job identifier.
        embed_base64 : bool
            If ``True``, embed image bytes as ``data:image/png;base64,...``
            strings in each image entry (useful for CLI / offline output).
        """
        sop_dict = sop.model_dump()
        image_base_url = f"/static/jobs/{job_id}/images"
        all_images: list[dict] = []

        for step in sop_dict["steps"]:
            step_num = step["step_number"]
            ts_list = step["evidence"]["recommended_screenshot_timestamps"]

            # Build / update images list for this step
            populated_images: list[dict] = []
            for img_idx, ts in enumerate(ts_list, start=1):
                image_id = f"step_{step_num}_img_{img_idx}"
                caption = _find_caption(step["images"], image_id, step["step_title"], ts)

                entry: dict = {"image_id": image_id, "caption": caption}
                if ts in frame_map:
                    fname = frame_map[ts].name
                    entry["url"] = f"{image_base_url}/{fname}"
                    if embed_base64:
                        entry["base64_data"] = _encode(frame_map[ts])
                else:
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
