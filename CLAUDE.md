# CLAUDE.md — sopgen

iFixit-internal tool that turns a screen recording into a structured SOP
(JSON + docx + bundled zip), powered by Gemini video understanding.
Single FastAPI app served by hypercorn in production, with a vanilla-JS
drag-drop frontend and a click-based CLI for batch / single-file runs.

## Run it

```bash
# Local dev (uvicorn + autoreload, includes Swagger /docs)
python -m sopgen.api.main

# Production-style (HTTP/2 / h2c, what the Dockerfile runs)
hypercorn sopgen.api.main:app --bind 0.0.0.0:8000

# CLI: single file
python -m sopgen run --video recording.mp4 --out ./out

# CLI: batch (drop videos in projects/recording/, then:)
python -m sopgen run

# Tests
python -m pytest
```

`SOPGEN_GEMINI_API_KEY` must be set (env or `.env`). All tests mock the
Gemini and ffmpeg layers, so they run offline.

## Layout

```
sopgen/
  api/             FastAPI app, routes, async pipeline runner,
                   in-memory JobRegistry, GuidesStats counter,
                   static/ (frontend + cowork_prompt.txt)
  core/            Settings, JobManager, ffmpeg wrapper, MIME validation,
                   SOPValidator + run_with_repair
  gemini/          google-genai client wrapper (streaming + HttpOptions
                   timeout), prompt builders, VideoAnalyzer
  render/          packager (json + image rename), docx_packager,
                   zip_packager (recursion-safe bundle + slugify)
  __main__.py      Click CLI (run --video / --folder)
tests/             pytest, all external services mocked
```

## Routing gotchas (these have bitten Claude before)

- **`/static` mounts `data_dir`, NOT the frontend dir.** The
  frontend mount is at `/`. So `sopgen/api/static/cowork_prompt.txt`
  is served at `/cowork_prompt.txt`, not `/static/cowork_prompt.txt`.
  Job images live under `data_dir/jobs/<id>/images/` and are reached
  via `/static/jobs/<id>/images/...`.
- **Mount order matters.** Routes are registered first, then `/static`,
  then `/` last (`html=True`). Don't reorder.

## TestClient pattern

Tests that exercise the async pipeline **must** use a context-managed
TestClient — starlette's per-request portal otherwise tears down before
the `asyncio.create_task` background work can resume past the first
`await`.

```python
@pytest.fixture()
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c
```

## Cross-cutting design choices

- **Substep upper bound is policy, not schema.** `SOPStep.substeps`
  carries only `min_length=1` on the Pydantic Field. The max is
  enforced by `SOPValidator(max_substeps=N)`, which `run_with_repair`
  builds from `analyzer.settings.max_substeps_per_step`. The same
  number is injected into the system / user / repair prompts so the
  model is told the same cap the validator enforces.
- **Frontend dark palette.** `--bg` (dark body) + `--surface` (elevated
  card) + `--fg` (light text) + `--accent` (cyan, only for selected /
  focused / dragover / primary buttons). `--surface-accent` is the OLD
  name and must not appear anywhere — there's a test that fails if it
  does. Button text on `--accent` uses `var(--bg)` for contrast (light
  `--fg` on bright cyan would be illegible).
- **Two button classes:** `.btn-primary` (Download) and `.primary-btn`
  (Copy Cowork prompt). Different selectors, both with `--accent` bg.
  Don't merge them.
- **Persistent guides counter / leaderboard** is process-local
  (`GuidesStats` with `threading.Lock`). At scale, swap for Firestore.
  Legacy `{"count": N}` files migrate cleanly on next increment.
- **Only successful API runs bump the counter.** The CLI does not
  increment. Errored runs do not increment.
- **Batch CLI output dirs are timestamp-suffixed**
  (`<stem>__YYYYMMDD-HHMMSS/`). Single-file `--out` is honored
  verbatim — controlled by the `timestamp_suffix` flag on
  `_process_video`.

## Windows quirks

- **`click.echo` with non-cp1252 chars crashes the CLI** when stdout is
  captured (e.g. by a background runner). Use `->` not `→` in user-
  facing print lines. There's no test for this; just don't add Unicode
  arrows to CLI output.
- **`ffmpeg` subprocess tests** skip cleanly when the sandbox blocks
  spawning. Total test count fluctuates between 134 and 137 — both are
  fine.

## Don'ts

- Don't add `git add .` — `.env` (real API key) and the local
  `.env.example` (also real) are gitignored / unstaged respectively.
  Use explicit `git add <paths>`.
- Don't commit `ab/`, `data/`, `projects/` — gitignored. They contain
  business video data.
- Don't include the literal U+2192 (`→`) in CLI prints.
- Don't break the `cowork_prompt.txt` formatting (the Cowork prompt is
  consumed by another LLM downstream; whitespace and prefixes matter).
- Don't make guides Public from the Cowork prompt or any tool. Private
  is the default and the only allowed state for now.
