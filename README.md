# SOP Generator

Generate structured Standard Operating Procedures from screen recording videos using the Google Gemini API.

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

**Prerequisites**: [ffmpeg](https://ffmpeg.org/download.html) must be installed and on your PATH.

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and set your Gemini API key:

```
SOPGEN_GEMINI_API_KEY=your_key_here
```

Get an API key at <https://aistudio.google.com/app/apikey>.

### 3. Run the API server

```bash
python -m sopgen.api.main
# → http://localhost:8000
# → Docs at http://localhost:8000/docs
```

### 4. Run via CLI

```bash
python -m sopgen run --video recording.mp4 --out ./output
```

Options:

| Flag | Description |
|------|-------------|
| `--video` | Path to screen recording (required) |
| `--out` | Output directory (required) |
| `--title-hint` | Suggested SOP title |
| `--domain-hint` | Domain context, e.g. `"NetSuite AP process"` |
| `--media-resolution` | `low` or `default` |
| `--fps-override` | Override default 1 FPS sampling |

## API Usage

### `POST /v1/sop`

Multipart form data:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `video` | file | yes | Screen recording video |
| `title_hint` | string | no | Suggested SOP title |
| `domain_hint` | string | no | Domain context |
| `media_resolution` | string | no | `"low"` or `"default"` |
| `fps_override` | int | no | Custom FPS for Gemini analysis |

**Example request:**

```bash
curl -X POST http://localhost:8000/v1/sop \
  -F "video=@recording.mp4" \
  -F "title_hint=Invoice Entry Process" \
  -F "domain_hint=NetSuite AP process"
```

**Example response:**

```json
{
  "job_id": "a1b2c3d4e5f6",
  "sop": {
    "title": "Invoice Entry Process",
    "intro": "This procedure covers entering a vendor invoice in NetSuite.",
    "settings": {
      "max_substeps_per_step": 4,
      "min_images_per_step": 1
    },
    "steps": [
      {
        "step_number": 1,
        "step_title": "Navigate to Enter Bills",
        "substeps": [
          "Open the NetSuite dashboard",
          "Click Transactions > Payables > Enter Bills"
        ],
        "evidence": {
          "recommended_screenshot_timestamps": ["00:05"],
          "supporting_timestamps": [
            {"start": "00:01", "end": "00:08", "why": "Navigation sequence"}
          ]
        },
        "images": [
          {"image_id": "step_1_img_1", "caption": "Enter Bills menu selection"}
        ]
      }
    ],
    "warnings": []
  },
  "image_base_url": "/static/jobs/a1b2c3d4e5f6/images",
  "images": [
    {
      "image_id": "step_1_img_1",
      "url": "/static/jobs/a1b2c3d4e5f6/images/frame_000_00_05.png",
      "caption": "Enter Bills menu selection"
    }
  ]
}
```

## How It Works

### Video Analysis Pipeline

1. **Upload** — Video is saved locally under `./data/uploads/`.
2. **Gemini Analysis** — Video is sent to the Gemini API with a structured prompt requesting SOP JSON output.
3. **Validation + Repair** — The JSON response is validated against a strict Pydantic schema. If constraints are violated (>4 substeps, missing screenshots), a repair prompt is sent automatically (up to 2 retries).
4. **Screenshot Extraction** — ffmpeg extracts frames at the timestamps Gemini recommended.
5. **Packaging** — The SOP JSON and images are merged into the final response.

### Inline vs Files API

The SDK automatically selects the upload method:

| Condition | Method | Why |
|-----------|--------|-----|
| File < 20 MB | **Inline data** | Simpler, single request |
| File >= 20 MB | **Files API upload** | Required for large files; supports polling for processing status |

The threshold is configurable via `SOPGEN_MAX_INLINE_SIZE_MB`.

### Timestamp Format

All timestamps use **MM:SS** format (e.g., `01:23` = 1 minute 23 seconds). Gemini samples video at approximately **1 frame per second** by default. You can override this with `fps_override`.

### Media Resolution

| Setting | Tokens | Best for |
|---------|--------|----------|
| `default` | Higher | Fine text, detailed UI elements |
| `low` | ~75% less | General workflow steps, faster processing |

Set via `SOPGEN_GEMINI_MEDIA_RESOLUTION` env var or `media_resolution` request parameter.

### Supported Video Formats

| Format | MIME Type |
|--------|-----------|
| MP4 | `video/mp4` |
| MPEG | `video/mpeg` |
| MOV | `video/mov`, `video/quicktime` |
| AVI | `video/avi` |
| FLV | `video/x-flv` |
| MPG | `video/mpg` |
| WebM | `video/webm` |
| WMV | `video/wmv` |
| 3GPP | `video/3gpp` |

## Configuration Reference

All settings use the `SOPGEN_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `SOPGEN_GEMINI_API_KEY` | *(required)* | Google AI API key |
| `SOPGEN_GEMINI_MODEL` | `gemini-2.0-flash` | Model to use |
| `SOPGEN_GEMINI_MEDIA_RESOLUTION` | `default` | `low` or `default` |
| `SOPGEN_GEMINI_VIDEO_FPS_OVERRIDE` | *(none)* | Custom FPS sampling |
| `SOPGEN_MAX_INLINE_SIZE_MB` | `20` | Inline/Files API threshold |
| `SOPGEN_MAX_RETRY_ATTEMPTS` | `2` | Validation repair retries |
| `SOPGEN_DATA_DIR` | `./data` | Local storage root |
| `SOPGEN_FFMPEG_PATH` | `ffmpeg` | Path to ffmpeg binary |

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

ffmpeg-dependent tests will be skipped automatically if ffmpeg is not installed.

## Data Retention Note

**MVP behavior**: Raw video uploads and extracted frames are stored locally under `./data/`. This directory may contain sensitive business process recordings. Implement appropriate access controls and retention policies for production use.

## Project Structure

```
sopgen/
  api/          FastAPI web service (routes, schemas)
  core/         Config, validation, ffmpeg, MIME detection
  gemini/       Gemini SDK client, prompt engineering
  render/       SOP + image packaging
tests/          Unit and smoke tests
```
