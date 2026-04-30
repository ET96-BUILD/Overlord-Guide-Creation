"""MIME type detection and video format validation."""

from __future__ import annotations

import mimetypes
from pathlib import Path

# Gemini-supported video MIME types.
SUPPORTED_VIDEO_MIMES: set[str] = {
    "video/mp4",
    "video/mpeg",
    "video/mov",
    "video/quicktime",
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/3gpp",
}

# Fallback: extension → MIME (mimetypes stdlib can miss some).
_EXT_MAP: dict[str, str] = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".mpg": "video/mpg",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
}


def detect_mime(file_path: Path) -> str:
    """Best-effort MIME detection from path (extension-based)."""
    mime, _ = mimetypes.guess_type(str(file_path))
    if mime and mime.startswith("video/"):
        return mime
    return _EXT_MAP.get(file_path.suffix.lower(), "application/octet-stream")


def is_supported_video(mime_type: str) -> bool:
    return mime_type in SUPPORTED_VIDEO_MIMES


def validate_video_file(file_path: Path) -> tuple[bool, str]:
    """Return ``(is_valid, detected_mime)``."""
    mime = detect_mime(file_path)
    return is_supported_video(mime), mime


def supported_types_list() -> list[str]:
    """Human-readable list for error messages."""
    return sorted(SUPPORTED_VIDEO_MIMES)
