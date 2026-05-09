"""Tests for the GeminiClient wrapper — focused on http timeout wiring
and the streaming generate_with_video path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sopgen.core.config import Settings
from sopgen.gemini.client import GeminiClient


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Ensure no SOPGEN_* env vars leak into the test (e.g. from .env)."""
    for var in (
        "SOPGEN_GEMINI_REQUEST_TIMEOUT_SECONDS",
        "SOPGEN_GEMINI_API_KEY",
        "SOPGEN_GEMINI_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


# ── HttpOptions wiring ──────────────────────────────────────────────────


class TestHttpOptionsTimeout:
    def test_timeout_uses_config_in_milliseconds(self):
        settings = Settings(
            gemini_api_key="x",
            gemini_request_timeout_seconds=300,
        )
        with patch("sopgen.gemini.client.genai.Client") as mock_client_cls:
            GeminiClient(settings)

        kwargs = mock_client_cls.call_args.kwargs
        assert kwargs["api_key"] == "x"

        http_options = kwargs["http_options"]
        # google-genai HttpOptions stores timeout in ms
        assert http_options.timeout == 300_000

    def test_default_timeout_is_600_seconds(self):
        settings = Settings(gemini_api_key="x")
        with patch("sopgen.gemini.client.genai.Client") as mock_client_cls:
            GeminiClient(settings)
        http_options = mock_client_cls.call_args.kwargs["http_options"]
        assert http_options.timeout == 600 * 1000


# ── Streaming generate_with_video ───────────────────────────────────────


class TestGenerateWithVideoStream:
    def _build_client(self) -> tuple[GeminiClient, MagicMock]:
        settings = Settings(gemini_api_key="x")
        with patch("sopgen.gemini.client.genai.Client") as mock_client_cls:
            mock_inner = MagicMock()
            mock_client_cls.return_value = mock_inner
            client = GeminiClient(settings)
        return client, mock_inner

    def test_stream_chunks_are_concatenated(self):
        client, inner = self._build_client()
        chunks = [
            SimpleNamespace(text="part one "),
            SimpleNamespace(text="part two "),
            SimpleNamespace(text=None),  # final chunk with no text
            SimpleNamespace(text="part three"),
        ]
        inner.models.generate_content_stream.return_value = iter(chunks)

        result = client.generate_with_video(
            video_part=MagicMock(),
            text_prompt="prompt",
        )

        assert result == "part one part two part three"
        inner.models.generate_content_stream.assert_called_once()
        # Non-streaming call must NOT be used in this path
        inner.models.generate_content.assert_not_called()

    def test_empty_stream_returns_empty_string(self):
        client, inner = self._build_client()
        inner.models.generate_content_stream.return_value = iter([])

        result = client.generate_with_video(
            video_part=MagicMock(),
            text_prompt="prompt",
        )
        assert result == ""
