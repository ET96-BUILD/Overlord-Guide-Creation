"""Orchestrates video → SOP JSON generation via Gemini.

Routes between inline-data and Files API based on file size, sends the
prompt, and extracts raw JSON from the response.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from sopgen.core.config import Settings
from sopgen.gemini.client import GeminiClient
from sopgen.gemini.prompts import (
    build_repair_prompt,
    build_sop_prompt,
    build_system_instruction,
)

logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """High-level entry point for video analysis."""

    def __init__(self, client: GeminiClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(
        self,
        video_path: Path,
        mime_type: str,
        *,
        title_hint: Optional[str] = None,
        domain_hint: Optional[str] = None,
        fps_override: Optional[int] = None,
    ) -> str:
        """Analyze a video file and return the raw SOP JSON string.

        Automatically selects inline vs Files API based on file size.
        """
        file_size = video_path.stat().st_size
        use_files_api = file_size > self.settings.max_inline_bytes

        if use_files_api:
            logger.info(
                "File size %d MB > threshold %d MB — using Files API",
                file_size // (1024 * 1024),
                self.settings.max_inline_size_mb,
            )
            file_obj = self.client.upload_video(video_path)
            video_part = self.client.build_file_part(file_obj)
        else:
            logger.info(
                "File size %d KB — using inline data",
                file_size // 1024,
            )
            video_bytes = video_path.read_bytes()
            video_part = self.client.build_inline_part(
                video_bytes, mime_type, fps_override=fps_override
            )

        prompt = build_sop_prompt(
            self.settings,
            title_hint=title_hint,
            domain_hint=domain_hint,
        )
        raw = self.client.generate_with_video(
            video_part,
            prompt,
            system_instruction=build_system_instruction(self.settings),
        )
        return _extract_json(raw)

    def repair(self, original_json: str, errors: list[str]) -> str:
        """Send a text-only repair prompt and return corrected JSON."""
        prompt = build_repair_prompt(original_json, errors, self.settings)
        raw = self.client.generate_text(prompt)
        return _extract_json(raw)


# ── Helpers ─────────────────────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Strip optional markdown fences from the model response."""
    text = text.strip()
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text
