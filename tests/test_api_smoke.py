"""Smoke tests for the FastAPI application."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from sopgen.api.main import create_app
from sopgen.api.schemas import SOPDocument
from sopgen.core.config import Settings


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        gemini_api_key="test-key",
        data_dir=tmp_path / "data",
    )
    app = create_app(settings)
    return TestClient(app)


# ── MIME rejection ──────────────────────────────────────────────────────


class TestMIMEValidation:
    def test_txt_file_rejected(self, client):
        files = {"video": ("doc.txt", io.BytesIO(b"hello"), "text/plain")}
        resp = client.post("/v1/sop", files=files)
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_json_file_rejected(self, client):
        files = {"video": ("data.json", io.BytesIO(b"{}"), "application/json")}
        resp = client.post("/v1/sop", files=files)
        assert resp.status_code == 400


# ── Happy path (mocked Gemini + ffmpeg) ─────────────────────────────────


def _fake_sop() -> SOPDocument:
    return SOPDocument.model_validate(
        {
            "title": "Test",
            "intro": "Intro text",
            "settings": {"max_substeps_per_step": 4, "min_images_per_step": 1},
            "steps": [
                {
                    "step_number": 1,
                    "step_title": "Step one",
                    "substeps": ["Do thing"],
                    "evidence": {
                        "recommended_screenshot_timestamps": ["00:02"],
                        "supporting_timestamps": [],
                    },
                    "images": [
                        {"image_id": "step_1_img_1", "caption": "Caption"}
                    ],
                }
            ],
            "warnings": [],
        }
    )


class TestHappyPath:
    @patch("sopgen.api.routes.FFmpegExtractor")
    @patch("sopgen.api.routes.run_with_repair")
    @patch("sopgen.api.routes.GeminiClient")
    @patch("sopgen.api.routes.VideoAnalyzer")
    def test_success(
        self,
        mock_analyzer_cls,
        mock_gemini_cls,
        mock_repair,
        mock_ffmpeg_cls,
        client,
        tmp_path,
    ):
        mock_repair.return_value = _fake_sop()
        mock_ffmpeg_cls.return_value.extract_all.return_value = {}

        # Create a tiny mp4-ish file so MIME detection works
        video_bytes = b"\x00" * 100
        files = {"video": ("test.mp4", io.BytesIO(video_bytes), "video/mp4")}
        resp = client.post("/v1/sop", files=files)

        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        assert body["sop"]["title"] == "Test"
        assert len(body["sop"]["steps"]) == 1
