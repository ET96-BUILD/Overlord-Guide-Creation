"""Smoke tests for the FastAPI application — async-with-polling flow."""

from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from sopgen.api.main import create_app
from sopgen.api.schemas import SOPDocument
from sopgen.core.config import Settings
from sopgen.core.jobs import JobManager


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def settings(tmp_path):
    # Each test gets its own data_dir AND its own stats_path so the
    # guides-created counter starts at 0 and never leaks between tests.
    return Settings(
        gemini_api_key="test-key",
        data_dir=tmp_path / "data",
        stats_path=tmp_path / "stats.json",
    )


@pytest.fixture()
def client(settings):
    """Long-lived TestClient. Using ``with`` keeps starlette's blocking
    portal alive across requests so background tasks created by
    ``asyncio.create_task`` continue to run between polls."""
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


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


def _wait_for_status(client, job_id, target, *, timeout=5.0) -> dict:
    """Poll /v1/jobs/<id>/status until *target* status is reached."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/v1/jobs/{job_id}/status")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["status"] == target:
            return last
        if last["status"] == "error" and target != "error":
            pytest.fail(f"job errored unexpectedly: {last.get('error')}")
        time.sleep(0.05)
    pytest.fail(f"job {job_id} never reached status={target!r}; last={last!r}")


def _patches():
    """Convenience: patch the four blocking pipeline boundaries."""
    return (
        patch("sopgen.api.pipeline.run_with_repair"),
        patch("sopgen.api.pipeline.GeminiClient"),
        patch("sopgen.api.pipeline.VideoAnalyzer"),
        patch("sopgen.api.pipeline.FFmpegExtractor"),
    )


# ═══════════════════════════════════════════════════════════════════════
#  MIME validation (sync, before the pipeline is dispatched)
# ═══════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════
#  Async happy path: POST → poll status → result + zip
# ═══════════════════════════════════════════════════════════════════════


class TestAsyncFlow:
    def test_post_returns_202_then_full_lifecycle(self, client):
        p_repair, p_gemini, p_analyzer, p_ffmpeg = _patches()
        with p_repair as mock_repair, p_gemini, p_analyzer, p_ffmpeg as mock_ff:
            mock_repair.return_value = _fake_sop()
            mock_ff.return_value.extract_all.return_value = {}

            # POST returns 202 + queued job
            files = {"video": ("test.mp4", io.BytesIO(b"\x00" * 100), "video/mp4")}
            resp = client.post("/v1/sop", files=files)
            assert resp.status_code == 202
            body = resp.json()
            assert body["status"] == "queued"
            assert "job_id" in body
            job_id = body["job_id"]

            # Poll until done (mocked pipeline runs essentially instantly)
            final = _wait_for_status(client, job_id, "done")
            assert final["stage"] == "done"
            assert final["error"] is None
            assert final["has_zip"] is True

            # /result returns the SOP JSON written to disk
            r = client.get(f"/v1/jobs/{job_id}/result")
            assert r.status_code == 200
            assert r.json()["sop"]["title"] == "Test"

            # /zip streams the bundle
            z = client.get(f"/v1/jobs/{job_id}/zip")
            assert z.status_code == 200
            assert z.headers["content-type"] == "application/zip"
            with zipfile.ZipFile(io.BytesIO(z.content)) as zf:
                names = set(zf.namelist())
            assert "sop.json" in names
            assert "sop.docx" in names


# ═══════════════════════════════════════════════════════════════════════
#  /v1/jobs/<id>/result error states
# ═══════════════════════════════════════════════════════════════════════


class TestJobResultEndpoint:
    def test_unknown_job_returns_404(self, client):
        r = client.get("/v1/jobs/no-such-job/result")
        assert r.status_code == 404

    def test_in_flight_returns_425(self, client):
        """Block the pipeline so the job stays in 'running' state, then
        confirm /result returns 425 Too Early."""
        block = threading.Event()

        def blocking_repair(*args, **kwargs):
            block.wait(timeout=5)
            return _fake_sop()

        p_repair, p_gemini, p_analyzer, p_ffmpeg = _patches()
        with p_repair as mock_repair, p_gemini, p_analyzer, p_ffmpeg as mock_ff:
            mock_repair.side_effect = blocking_repair
            mock_ff.return_value.extract_all.return_value = {}

            files = {"video": ("x.mp4", io.BytesIO(b"\x00"), "video/mp4")}
            resp = client.post("/v1/sop", files=files)
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            # Wait for the pipeline to actually start (status='running')
            _wait_for_status(client, job_id, "running", timeout=2.0)

            r = client.get(f"/v1/jobs/{job_id}/result")
            assert r.status_code == 425
            assert "not done yet" in r.json()["detail"].lower()

            # Unblock so the test cleanup doesn't have a dangling worker
            block.set()
            _wait_for_status(client, job_id, "done", timeout=5.0)

    def test_errored_job_returns_500(self, client):
        p_repair, p_gemini, p_analyzer, p_ffmpeg = _patches()
        with p_repair as mock_repair, p_gemini, p_analyzer, p_ffmpeg as mock_ff:
            mock_repair.side_effect = RuntimeError("simulated boom")
            mock_ff.return_value.extract_all.return_value = {}

            files = {"video": ("x.mp4", io.BytesIO(b"\x00"), "video/mp4")}
            resp = client.post("/v1/sop", files=files)
            job_id = resp.json()["job_id"]

            _wait_for_status(client, job_id, "error", timeout=3.0)

            r = client.get(f"/v1/jobs/{job_id}/result")
            assert r.status_code == 500
            assert "boom" in r.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════
#  /v1/jobs/<id>/status edge cases
# ═══════════════════════════════════════════════════════════════════════


class TestJobStatusEndpoint:
    def test_unknown_job_returns_404(self, client):
        r = client.get("/v1/jobs/nope/status")
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════════════
#  /v1/jobs/<id>/zip — pre-existing flow, still works
# ═══════════════════════════════════════════════════════════════════════


def _seed_job_dir(settings: Settings, job_id: str, *, title: str | None) -> None:
    """Pre-populate a fake completed-job directory under settings.data_dir."""
    jobs = JobManager(settings)
    jobs._ensure_dirs(job_id)

    if title is not None:
        sop_payload = {
            "job_id": job_id,
            "sop": {
                "title": title,
                "intro": "Intro.",
                "settings": {"max_substeps_per_step": 4, "min_images_per_step": 1},
                "steps": [],
                "warnings": [],
            },
            "image_base_url": f"/static/jobs/{job_id}/images",
            "images": [],
        }
        jobs.sop_json_path(job_id).write_text(
            json.dumps(sop_payload), encoding="utf-8"
        )

    (jobs.job_dir(job_id) / "sop.docx").write_bytes(b"fake docx body")
    (jobs.images_dir(job_id) / "step_1_img_1.png").write_bytes(b"\x89PNG fake")


class TestJobZipDownload:
    def test_returns_zip_with_slugified_title_filename(self, client, settings):
        job_id = "abc123def456"
        _seed_job_dir(settings, job_id, title="How to Pay EAL Commission")

        resp = client.get(f"/v1/jobs/{job_id}/zip")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        cd = resp.headers["content-disposition"]
        assert 'filename="how-to-pay-eal-commission.zip"' in cd

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = set(zf.namelist())
        assert "sop.json" in names
        assert "sop.docx" in names
        assert "images/step_1_img_1.png" in names
        assert "sop_bundle.zip" not in names

    def test_falls_back_to_job_id_when_no_title(self, client, settings):
        job_id = "deadbeef0001"
        _seed_job_dir(settings, job_id, title=None)

        resp = client.get(f"/v1/jobs/{job_id}/zip")
        assert resp.status_code == 200
        cd = resp.headers["content-disposition"]
        assert f'filename="sop_{job_id}.zip"' in cd

    def test_unknown_job_returns_404(self, client):
        resp = client.get("/v1/jobs/no-such-job/zip")
        assert resp.status_code == 404
        assert "no-such-job" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════
#  Model selector — POST override + GET /v1/config
# ═══════════════════════════════════════════════════════════════════════


class TestConfigEndpoint:
    def test_returns_configured_default_model(self, client, settings):
        resp = client.get("/v1/config")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"default_model": settings.gemini_model}

    def test_default_model_tracks_settings_override(self, settings):
        """If the app is built with a non-default gemini_model, /v1/config
        reflects that — it reads from app.state.settings, not from env."""
        from sopgen.api.main import create_app

        custom = settings.model_copy(update={"gemini_model": "gemini-2.5-flash"})
        app = create_app(custom)
        with TestClient(app) as c:
            resp = c.get("/v1/config")
        assert resp.status_code == 200
        assert resp.json()["default_model"] == "gemini-2.5-flash"


class TestModelOverride:
    def test_post_with_flash_override_uses_that_model(self, client):
        p_repair, p_gemini, p_analyzer, p_ffmpeg = _patches()
        with p_repair as mock_repair, p_gemini as mock_gemini_cls, p_analyzer, p_ffmpeg as mock_ff:
            mock_repair.return_value = _fake_sop()
            mock_ff.return_value.extract_all.return_value = {}

            files = {"video": ("x.mp4", io.BytesIO(b"\x00" * 100), "video/mp4")}
            data = {"model": "gemini-2.5-flash"}
            resp = client.post("/v1/sop", files=files, data=data)
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            _wait_for_status(client, job_id, "done")

            # GeminiClient was constructed with the overridden model.
            assert mock_gemini_cls.called
            passed_settings = mock_gemini_cls.call_args.args[0]
            assert passed_settings.gemini_model == "gemini-2.5-flash"

    def test_post_with_invalid_model_returns_400(self, client):
        files = {"video": ("x.mp4", io.BytesIO(b"\x00"), "video/mp4")}
        data = {"model": "gemini-2.0-flash"}  # retired, not on whitelist
        resp = client.post("/v1/sop", files=files, data=data)
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        # Error message names the allowed values so the caller can self-correct
        assert "gemini-2.5-pro" in detail
        assert "gemini-2.5-flash" in detail
        assert "gemini-2.0-flash" in detail  # echoes back what was rejected

    def test_post_without_model_uses_configured_default(self, client, settings):
        p_repair, p_gemini, p_analyzer, p_ffmpeg = _patches()
        with p_repair as mock_repair, p_gemini as mock_gemini_cls, p_analyzer, p_ffmpeg as mock_ff:
            mock_repair.return_value = _fake_sop()
            mock_ff.return_value.extract_all.return_value = {}

            files = {"video": ("x.mp4", io.BytesIO(b"\x00" * 100), "video/mp4")}
            resp = client.post("/v1/sop", files=files)  # no `model` field
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            _wait_for_status(client, job_id, "done")

            passed_settings = mock_gemini_cls.call_args.args[0]
            assert passed_settings.gemini_model == settings.gemini_model


# ═══════════════════════════════════════════════════════════════════════
#  Static frontend at "/"
# ═══════════════════════════════════════════════════════════════════════


class TestStaticFrontend:
    def test_root_serves_index_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "iFixit SOP Generator" in r.text

    def test_index_html_references_v1_endpoints(self, client):
        r = client.get("/")
        body = r.text
        # The frontend must POST to /v1/sop and poll /v1/jobs/<id>/status
        assert "/v1/sop" in body
        assert "/v1/jobs/" in body

    def test_index_html_has_model_radios(self, client):
        r = client.get("/")
        body = r.text
        # Both model values must be present as radio options
        assert 'value="gemini-2.5-pro"' in body
        assert 'value="gemini-2.5-flash"' in body
        # Frontend pre-selects from /v1/config
        assert "/v1/config" in body
        # Helper copy from spec
        assert "If you are unhappy with the output, try another model." in body

    def test_index_html_uses_dark_mode_palette(self, client):
        body = client.get("/").text
        # All variables must be defined at :root with the exact spec values.
        for decl in (
            "--bg: #454447",         # dark body
            "--surface: #5a5a5d",    # elevated dark card surface
            "--fg: #f5f5f7",         # light text
            "--fg-on-bg: #f5f5f7",   # alias of --fg
            "--fg-muted: #B5B5BA",
            "--accent: #67D7F8",
            "--error: #ff7676",
        ):
            assert decl in body, f"missing CSS variable: {decl!r}"

    def test_index_html_has_no_stale_surface_accent_var(self, client):
        """The --surface-accent variable was renamed to --surface; any
        stray reference would silently fall through to the browser's
        default and break the dark theme."""
        body = client.get("/").text
        assert "--surface-accent" not in body

    def test_index_html_has_dragover_styles(self, client):
        body = client.get("/").text
        # The dragover class must have its own dedicated rule, separate
        # from the static :hover rule.
        assert ".dropzone.dragover" in body
        # Glow alpha now 0.20 (was 0.16) since contrast against the
        # darker --surface is higher than against bright cyan.
        assert "rgba(103, 215, 248, 0.20)" in body

    def test_index_html_has_leaderboard_sidebar(self, client):
        body = client.get("/").text
        assert 'id="sidebar"' in body
        assert 'id="leaderboard"' in body
        assert "/v1/leaderboard" in body
        assert "Top guide makers" in body

    def test_sidebar_has_two_stacked_cards(self, client):
        body = client.get("/").text
        # Two sidebar-card sections: "Top guide makers" + "Guides created"
        assert body.count("sidebar-card") >= 2
        assert "Top guide makers" in body
        assert "Guides created" in body

    def test_stats_total_lives_in_sidebar_card(self, client):
        body = client.get("/").text
        # The total now lives in the sidebar, not in a header badge.
        assert 'id="stats-total"' in body
        assert "/v1/stats" in body

    def test_old_stats_badge_is_removed_from_header(self, client):
        body = client.get("/").text
        assert 'id="stats-badge"' not in body
        # The old badge text format ("Guides created: N") shouldn't render
        # anymore — the new card uses just "Guides created" as a heading.
        assert "Guides created:" not in body

    def test_title_hint_placeholder_is_generic(self, client):
        body = client.get("/").text
        assert "How to Pay Vendor Commission" in body
        assert "How to Pay EAL Commission" not in body


# ═══════════════════════════════════════════════════════════════════════
#  Cowork publish-help: prompt file is reachable + Done panel markup
# ═══════════════════════════════════════════════════════════════════════


class TestCoworkPrompt:
    def test_cowork_prompt_txt_is_served(self, client):
        # The frontend dir is mounted at "/", so the canonical prompt is
        # reachable at /cowork_prompt.txt. (The /static mount serves job
        # artifacts from data_dir, not app assets.)
        resp = client.get("/cowork_prompt.txt")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        body = resp.text
        # The leading hash header from the spec must be intact.
        assert "# iFixit Guide Builder" in body
        # Spot-check a few canonical strings that drive Cowork behaviour.
        assert "Bulk Step Creator" in body
        assert "Caution:" in body
        assert "Begin by confirming the two inputs." in body

    def test_done_panel_has_publish_help_markup(self, client):
        body = client.get("/").text
        assert "Publish to Overlord with Cowork" in body
        assert '<ol class="publish-steps">' in body
        assert 'id="copy-prompt"' in body
        assert 'id="copy-prompt-status"' in body

    def test_publish_help_reuses_sidebar_card_class(self, client):
        body = client.get("/").text
        # The publish-help section is supposed to inherit the sidebar
        # light-card chrome via shared classes.
        assert 'class="publish-help sidebar-card"' in body

    def test_index_html_references_cowork_prompt_url(self, client):
        body = client.get("/").text
        # The JS must reference the prompt URL so the click handler can
        # fetch it.
        assert "/cowork_prompt.txt" in body


# ═══════════════════════════════════════════════════════════════════════
#  Guides-created counter — endpoint + pipeline integration
# ═══════════════════════════════════════════════════════════════════════


class TestStatsEndpoint:
    def test_returns_zero_initially(self, client):
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        assert resp.json() == {"guides_created": 0}

    def test_reflects_in_memory_increments(self, client):
        # Bump the counter directly via the app-state stats object.
        client.app.state.stats.increment()
        client.app.state.stats.increment()

        resp = client.get("/v1/stats")
        assert resp.json() == {"guides_created": 2}


class TestPipelineCounterIntegration:
    def test_successful_run_increments_counter(self, client):
        assert client.get("/v1/stats").json()["guides_created"] == 0

        p_repair, p_gemini, p_analyzer, p_ffmpeg = _patches()
        with p_repair as mock_repair, p_gemini, p_analyzer, p_ffmpeg as mock_ff:
            mock_repair.return_value = _fake_sop()
            mock_ff.return_value.extract_all.return_value = {}

            files = {"video": ("ok.mp4", io.BytesIO(b"\x00" * 100), "video/mp4")}
            resp = client.post("/v1/sop", files=files)
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            _wait_for_status(client, job_id, "done")

        assert client.get("/v1/stats").json()["guides_created"] == 1

    def test_errored_run_does_not_increment_counter(self, client):
        assert client.get("/v1/stats").json()["guides_created"] == 0

        p_repair, p_gemini, p_analyzer, p_ffmpeg = _patches()
        with p_repair as mock_repair, p_gemini, p_analyzer, p_ffmpeg as mock_ff:
            mock_repair.side_effect = RuntimeError("simulated boom")
            mock_ff.return_value.extract_all.return_value = {}

            files = {"video": ("bad.mp4", io.BytesIO(b"\x00"), "video/mp4")}
            resp = client.post("/v1/sop", files=files)
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            _wait_for_status(client, job_id, "error")

        # Counter must NOT have moved.
        assert client.get("/v1/stats").json()["guides_created"] == 0

    def test_invalid_mime_does_not_increment_counter(self, client):
        # The MIME-rejection path returns 400 synchronously before the
        # pipeline is even dispatched — counter must stay at 0.
        files = {"video": ("doc.txt", io.BytesIO(b"hi"), "text/plain")}
        resp = client.post("/v1/sop", files=files)
        assert resp.status_code == 400

        assert client.get("/v1/stats").json()["guides_created"] == 0


# ═══════════════════════════════════════════════════════════════════════
#  Leaderboard endpoint + IAP email passthrough
# ═══════════════════════════════════════════════════════════════════════


class TestLeaderboardEndpoint:
    def test_returns_empty_top_initially(self, client):
        resp = client.get("/v1/leaderboard")
        assert resp.status_code == 200
        assert resp.json() == {"top": []}

    def test_reflects_in_memory_increments(self, client):
        client.app.state.stats.increment(email="alice@ifixit.com")
        client.app.state.stats.increment(email="alice@ifixit.com")
        client.app.state.stats.increment(email="bob@ifixit.com")

        resp = client.get("/v1/leaderboard")
        assert resp.status_code == 200
        assert resp.json() == {
            "top": [
                {"email": "alice@ifixit.com", "count": 2},
                {"email": "bob@ifixit.com", "count": 1},
            ]
        }

    def test_limit_query_param_is_honored(self, client):
        for _ in range(3): client.app.state.stats.increment(email="a@x.com")
        for _ in range(2): client.app.state.stats.increment(email="b@x.com")
        client.app.state.stats.increment(email="c@x.com")

        resp = client.get("/v1/leaderboard?limit=2")
        assert resp.status_code == 200
        emails = [e["email"] for e in resp.json()["top"]]
        assert emails == ["a@x.com", "b@x.com"]


class TestIAPEmailPassthrough:
    def test_iap_header_attributes_run_to_user(self, client):
        p_repair, p_gemini, p_analyzer, p_ffmpeg = _patches()
        with p_repair as mock_repair, p_gemini, p_analyzer, p_ffmpeg as mock_ff:
            mock_repair.return_value = _fake_sop()
            mock_ff.return_value.extract_all.return_value = {}

            files = {"video": ("ok.mp4", io.BytesIO(b"\x00" * 100), "video/mp4")}
            headers = {
                "X-Goog-Authenticated-User-Email": "accounts.google.com:emerson@ifixit.com",
            }
            resp = client.post("/v1/sop", files=files, headers=headers)
            assert resp.status_code == 202
            job_id = resp.json()["job_id"]

            _wait_for_status(client, job_id, "done")

        # Total bumped by 1 AND user_email was attributed without the
        # "accounts.google.com:" prefix.
        assert client.get("/v1/stats").json()["guides_created"] == 1
        board = client.get("/v1/leaderboard").json()["top"]
        assert board == [{"email": "emerson@ifixit.com", "count": 1}]

    def test_no_iap_header_runs_anonymously(self, client):
        p_repair, p_gemini, p_analyzer, p_ffmpeg = _patches()
        with p_repair as mock_repair, p_gemini, p_analyzer, p_ffmpeg as mock_ff:
            mock_repair.return_value = _fake_sop()
            mock_ff.return_value.extract_all.return_value = {}

            files = {"video": ("ok.mp4", io.BytesIO(b"\x00" * 100), "video/mp4")}
            resp = client.post("/v1/sop", files=files)  # no IAP header
            job_id = resp.json()["job_id"]
            _wait_for_status(client, job_id, "done")

        # Total bumped, leaderboard stays empty (anonymous run).
        assert client.get("/v1/stats").json()["guides_created"] == 1
        assert client.get("/v1/leaderboard").json()["top"] == []

    def test_iap_header_without_prefix_still_works(self, client):
        """Some dev environments send the bare email without the
        "accounts.google.com:" prefix — accept it as-is."""
        p_repair, p_gemini, p_analyzer, p_ffmpeg = _patches()
        with p_repair as mock_repair, p_gemini, p_analyzer, p_ffmpeg as mock_ff:
            mock_repair.return_value = _fake_sop()
            mock_ff.return_value.extract_all.return_value = {}

            files = {"video": ("ok.mp4", io.BytesIO(b"\x00" * 100), "video/mp4")}
            headers = {"X-Goog-Authenticated-User-Email": "dev@ifixit.com"}
            resp = client.post("/v1/sop", files=files, headers=headers)
            job_id = resp.json()["job_id"]
            _wait_for_status(client, job_id, "done")

        board = client.get("/v1/leaderboard").json()["top"]
        assert board == [{"email": "dev@ifixit.com", "count": 1}]
