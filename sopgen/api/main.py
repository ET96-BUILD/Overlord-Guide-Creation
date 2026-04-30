"""FastAPI application factory and uvicorn entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from sopgen.api.routes import router
from sopgen.core.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    app = FastAPI(
        title="SOP Generator",
        description="Generate structured SOPs from screen recording videos via Gemini API.",
        version="0.1.0",
    )
    app.state.settings = settings

    # Serve extracted images at /static/jobs/{job_id}/images/…
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(data_dir)), name="static")

    app.include_router(router, prefix="/v1")

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
