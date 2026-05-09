"""Thin wrapper around the google-genai SDK for video understanding."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from sopgen.core.config import Settings

logger = logging.getLogger(__name__)

_POLL_INTERVAL_S = 3
_MAX_POLL_S = 300  # 5 min


class GeminiClient:
    """Manages the GenAI client, video uploads, and content generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # google-genai HttpOptions.timeout is in milliseconds. The default
        # underlying httpx timeout (~60s) is too short for long video
        # inference, so set it explicitly from config.
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=settings.gemini_request_timeout_seconds * 1000,
            ),
        )
        self.model = settings.gemini_model

    # ── Files API upload ────────────────────────────────────────────────

    def upload_video(self, video_path: Path) -> object:
        """Upload via Files API and poll until processing completes.

        Returns the file object that can be passed to ``generate_content``.
        """
        logger.info("Uploading video via Files API: %s", video_path.name)
        file_obj = self._client.files.upload(file=str(video_path))
        logger.info("Upload started — name=%s", file_obj.name)

        elapsed = 0.0
        while getattr(file_obj, "state", None) == "PROCESSING":
            if elapsed >= _MAX_POLL_S:
                raise TimeoutError(
                    f"Video still processing after {_MAX_POLL_S}s: {file_obj.name}"
                )
            time.sleep(_POLL_INTERVAL_S)
            elapsed += _POLL_INTERVAL_S
            file_obj = self._client.files.get(name=file_obj.name)
            logger.debug("Poll — state=%s elapsed=%.0fs", file_obj.state, elapsed)

        if getattr(file_obj, "state", None) == "FAILED":
            raise RuntimeError(f"Files API processing failed for {file_obj.name}")

        logger.info("Video ready: %s", file_obj.uri)
        return file_obj

    # ── Generation ──────────────────────────────────────────────────────

    def generate_with_video(
        self,
        video_part: types.Part,
        text_prompt: str,
        *,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Send video + text to the model and return the raw text response.

        Uses streaming so the connection stays alive while the model is
        producing tokens — long video inference can otherwise blow past
        httpx's default idle timeout before the first byte arrives.
        """
        config = self._build_config(system_instruction)
        contents: list = [video_part, text_prompt]

        logger.info("Generating with video (stream) — model=%s", self.model)
        chunks: list[str] = []
        stream = self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                chunks.append(text)
        return "".join(chunks)

    def generate_text(self, prompt: str) -> str:
        """Text-only generation (used for repair prompts)."""
        logger.info("Generating text-only — model=%s", self.model)
        response = self._client.models.generate_content(
            model=self.model,
            contents=[prompt],
        )
        return response.text

    # ── Part builders ───────────────────────────────────────────────────

    def build_inline_part(
        self,
        video_bytes: bytes,
        mime_type: str,
        fps_override: Optional[int] = None,
    ) -> types.Part:
        """Create an inline video Part (for small files)."""
        blob = types.Blob(data=video_bytes, mime_type=mime_type)
        kwargs: dict = {"inline_data": blob}
        fps = fps_override or self.settings.gemini_video_fps_override
        if fps is not None:
            kwargs["video_metadata"] = types.VideoMetadata(fps=fps)
        return types.Part(**kwargs)

    def build_file_part(self, file_obj: object) -> types.Part:
        """Create a Part referencing an uploaded Files API object."""
        return types.Part(
            file_data=types.FileData(
                file_uri=file_obj.uri,
                mime_type=getattr(file_obj, "mime_type", "video/mp4"),
            )
        )

    # ── Internal ────────────────────────────────────────────────────────

    def _build_config(
        self, system_instruction: Optional[str] = None
    ) -> types.GenerateContentConfig:
        res = self.settings.gemini_media_resolution
        media_res = (
            types.MediaResolution.MEDIA_RESOLUTION_LOW
            if res == "low"
            else types.MediaResolution.MEDIA_RESOLUTION_HIGH
        )
        cfg_kwargs: dict = {"media_resolution": media_res}
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        return types.GenerateContentConfig(**cfg_kwargs)
