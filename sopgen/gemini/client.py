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

# Backoff growth factor for 503 retries. With base=5 and 3 attempts the
# schedule becomes 5s + 15s = 20s wait — quick enough that a transient
# load spike resolves before the user gives up, slow enough that we
# aren't hammering an overloaded model.
_UNAVAILABLE_BACKOFF_FACTOR = 3


def _is_unavailable_error(exc: BaseException) -> bool:
    """True iff *exc* looks like a Gemini 503 / UNAVAILABLE response.

    google-genai surfaces 503s as ``ServerError`` (or ``APIError``) with
    ``code=503``. We also string-match the message because SDK versions
    differ and the same condition occasionally arrives as a wrapped
    error without a populated ``code`` attribute.
    """
    code = getattr(exc, "code", None)
    if code == 503:
        return True
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status == 503:
        return True
    msg = str(exc).lower()
    return (
        "unavailable" in msg
        or "503" in msg
        or "high demand" in msg
        or "overloaded" in msg
    )


def _call_with_unavailable_retry(
    fn,
    *,
    max_attempts: int,
    base_delay_seconds: int,
):
    """Run *fn* and retry on 503 with backoff ``base * 3 ** attempt``.

    Only 503/UNAVAILABLE retries — anything else (400, 401, 429, 500,
    network) propagates immediately. After *max_attempts*, the most
    recent error is re-raised so the caller still sees the original
    exception type and traceback context.

    Uses ``time.sleep`` at call time (not via a default arg) so tests
    can ``patch("sopgen.gemini.client.time.sleep")`` to skip the real
    20-second wait.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — we classify below
            if not _is_unavailable_error(exc):
                raise
            last_err = exc
            if attempt >= max_attempts - 1:
                break
            delay = base_delay_seconds * (_UNAVAILABLE_BACKOFF_FACTOR ** attempt)
            logger.info(
                "Gemini returned 503 (high demand) — retrying in %ds "
                "(attempt %d/%d)",
                delay, attempt + 2, max_attempts,
            )
            time.sleep(delay)
    assert last_err is not None  # defensive — loop only exits via raise or return
    raise last_err


def _format_file_error(file_obj: object) -> str:
    """Pull whatever Google attached to a FAILED file into a readable string.

    The google-genai SDK exposes the failure detail as ``file.error`` with
    ``code`` and ``message`` fields, but older/newer SDK versions and edge
    cases sometimes leave it as None or a dict. Be defensive — we never
    want to mask the original error with an AttributeError.
    """
    err = getattr(file_obj, "error", None)
    if err is None:
        return "no error detail returned"
    message = getattr(err, "message", None) or (
        err.get("message") if isinstance(err, dict) else None
    )
    code = getattr(err, "code", None) or (
        err.get("code") if isinstance(err, dict) else None
    )
    if message and code is not None:
        return f"{message} (code={code})"
    return str(message or err)


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
        Raises ``RuntimeError`` with Google's status detail on FAILED state.
        """
        size_bytes = video_path.stat().st_size
        logger.info(
            "Uploading video via Files API: %s (%.1f MB)",
            video_path.name,
            size_bytes / (1024 * 1024),
        )
        file_obj = self._client.files.upload(file=str(video_path))
        logger.info(
            "Upload started — name=%s state=%s mime_type=%s",
            file_obj.name,
            getattr(file_obj, "state", None),
            getattr(file_obj, "mime_type", None),
        )

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
            reason = _format_file_error(file_obj)
            logger.error(
                "Files API processing FAILED for %s (%s): %s",
                file_obj.name,
                video_path.name,
                reason,
            )
            raise RuntimeError(
                f"Files API processing failed for {file_obj.name}: {reason}. "
                f"Most likely an unsupported codec, corrupt/truncated file, "
                f"or a renamed non-mp4 — try re-encoding to H.264 + AAC mp4."
            )

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
        # The 503 surfaces at stream establishment (the initial HTTP
        # request), not during iteration — so we only need to wrap this
        # call. Iteration over the stream is left untouched so the
        # streaming response semantics are preserved.
        stream = _call_with_unavailable_retry(
            lambda: self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            ),
            max_attempts=self.settings.gemini_unavailable_max_attempts,
            base_delay_seconds=self.settings.gemini_unavailable_base_delay_seconds,
        )
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                chunks.append(text)
        return "".join(chunks)

    def generate_text(self, prompt: str) -> str:
        """Text-only generation (used for repair prompts)."""
        logger.info("Generating text-only — model=%s", self.model)
        response = _call_with_unavailable_retry(
            lambda: self._client.models.generate_content(
                model=self.model,
                contents=[prompt],
            ),
            max_attempts=self.settings.gemini_unavailable_max_attempts,
            base_delay_seconds=self.settings.gemini_unavailable_base_delay_seconds,
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
