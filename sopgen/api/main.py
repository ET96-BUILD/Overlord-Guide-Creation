"""FastAPI application factory and uvicorn entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from sopgen.api.job_registry import JobRegistry
from sopgen.api.routes import router
from sopgen.api.stats import GuidesStats
from sopgen.core.config import Settings
from sopgen.core.jobs import JobManager


_FRONTEND_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="SOP Generator",
        description="Generate structured SOPs from screen recording videos via Gemini API.",
        version="0.1.0",
    )
    app.state.settings = settings
    # Disk-backed job registry: in-memory hot path + per-job
    # <job_dir>/status.json mirror so a container restart can recover
    # the state. Single-worker is still authoritative within a process;
    # at horizontal scale, swap for Redis/Firestore.
    app.state.job_registry = JobRegistry(jobs=JobManager(settings))
    # Persistent guides-created counter. Loaded from disk on first read.
    app.state.stats = GuidesStats(settings)

    # ── Routes ──────────────────────────────────────────────────────
    app.include_router(router, prefix="/v1")

    # ── Static images (jobs' rendered screenshots) ──────────────────
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(data_dir)), name="static")

    # ── Drag-drop frontend at "/" ───────────────────────────────────
    # Mounted LAST so /v1/* and /static/* routes match first; the
    # html=True flag makes "/" serve index.html directly.
    if _FRONTEND_DIR.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(_FRONTEND_DIR), html=True),
            name="frontend",
        )

    return app


# Default instance for ``uvicorn sopgen.api.main:app``
app = create_app()


if __name__ == "__main__":
    import uvicorn

    _settings = Settings()
    uvicorn.run(
        "sopgen.api.main:app",
        host=_settings.api_host,
        port=_settings.api_port,
        reload=True,
    )
