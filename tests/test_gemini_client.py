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
        "SOPGEN_GEMINI_UNAVAILABLE_MAX_ATTEMPTS",
        "SOPGEN_GEMINI_UNAVAILABLE_BASE_DELAY_SECONDS",
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


# ── Retry on 503 UNAVAILABLE ───────────────────────────────────────────


def _build_client_with_inner() -> tuple[GeminiClient, MagicMock]:
    settings = Settings(gemini_api_key="x")
    with patch("sopgen.gemini.client.genai.Client") as mock_client_cls:
        mock_inner = MagicMock()
        mock_client_cls.return_value = mock_inner
        client = GeminiClient(settings)
    return client, mock_inner


def _make_503(message: str = "503 UNAVAILABLE: The model is overloaded"):
    err = RuntimeError(message)
    err.code = 503  # mimic google-genai ServerError.code
    return err


class TestRetryOnUnavailable:
    def test_single_503_then_success_returns_result(self):
        client, inner = _build_client_with_inner()
        ok_chunks = [SimpleNamespace(text="recovered")]
        inner.models.generate_content_stream.side_effect = [
            _make_503(),
            iter(ok_chunks),
        ]

        with patch("sopgen.gemini.client.time.sleep") as mock_sleep:
            result = client.generate_with_video(
                video_part=MagicMock(), text_prompt="prompt"
            )

        assert result == "recovered"
        assert inner.models.generate_content_stream.call_count == 2
        # First retry sleeps for the base delay (5s by default).
        assert mock_sleep.call_args_list[0].args[0] == 5

    def test_three_503s_in_a_row_raises_original_error(self):
        client, inner = _build_client_with_inner()
        inner.models.generate_content_stream.side_effect = [
            _make_503("first"),
            _make_503("second"),
            _make_503("third"),
        ]

        with patch("sopgen.gemini.client.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="third"):
                client.generate_with_video(
                    video_part=MagicMock(), text_prompt="prompt"
                )

        # 3 attempts → 2 sleeps (5s then 15s).
        assert inner.models.generate_content_stream.call_count == 3
        assert [c.args[0] for c in mock_sleep.call_args_list] == [5, 15]

    def test_400_error_does_not_retry(self):
        """A 400 / 401 / 429 / 500 / network error is a real failure
        we want to surface immediately — not a model-overloaded blip."""
        client, inner = _build_client_with_inner()
        err = RuntimeError("400 INVALID_ARGUMENT: bad video")
        err.code = 400
        inner.models.generate_content_stream.side_effect = [err]

        with patch("sopgen.gemini.client.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
                client.generate_with_video(
                    video_part=MagicMock(), text_prompt="prompt"
                )

        assert inner.models.generate_content_stream.call_count == 1
        mock_sleep.assert_not_called()

    def test_429_rate_limit_does_not_retry(self):
        """429 is rate-limiting, not over-demand — surfaces immediately."""
        client, inner = _build_client_with_inner()
        err = RuntimeError("429 RESOURCE_EXHAUSTED")
        err.code = 429
        inner.models.generate_content_stream.side_effect = [err]

        with patch("sopgen.gemini.client.time.sleep") as mock_sleep:
            with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
                client.generate_with_video(
                    video_part=MagicMock(), text_prompt="prompt"
                )

        assert inner.models.generate_content_stream.call_count == 1
        mock_sleep.assert_not_called()

    def test_retry_also_applies_to_generate_text(self):
        """The repair-pass code path (text-only) gets the same treatment
        as the video-bearing path."""
        client, inner = _build_client_with_inner()
        success = SimpleNamespace(text="repaired")
        inner.models.generate_content.side_effect = [_make_503(), success]

        with patch("sopgen.gemini.client.time.sleep") as mock_sleep:
            result = client.generate_text("please fix this")

        assert result == "repaired"
        assert inner.models.generate_content.call_count == 2
        assert mock_sleep.call_args_list[0].args[0] == 5

    def test_string_match_classifies_unavailable_without_code_attr(self):
        """Older SDK wraps the upstream error in a way that loses the
        .code attribute. The string-match in _is_unavailable_error
        catches these too."""
        client, inner = _build_client_with_inner()
        err = RuntimeError("ServiceUnavailable: The model is overloaded")
        # NOTE: no .code attribute set
        ok_chunks = [SimpleNamespace(text="ok")]
        inner.models.generate_content_stream.side_effect = [err, iter(ok_chunks)]

        with patch("sopgen.gemini.client.time.sleep"):
            result = client.generate_with_video(
                video_part=MagicMock(), text_prompt="prompt"
            )

        assert result == "ok"
        assert inner.models.generate_content_stream.call_count == 2
