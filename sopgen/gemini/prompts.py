"""Prompt templates for Gemini video analysis and SOP generation."""

from __future__ import annotations

from typing import Optional

# ═══════════════════════════════════════════════════════════════════════
#  System instruction (passed via GenerateContentConfig)
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_INSTRUCTION = """\
You are an expert technical writer generating Standard Operating Procedure (SOP) \
documentation from screen recordings of business processes.

Rules:
- Write concise, imperative substeps (e.g. "Click Save", "Enter the vendor name").
- Assume the UI resembles NetSuite or similar ERP/business software unless the \
  recording clearly shows otherwise.
- Each step MUST have <= 4 substeps.
- Each step MUST have >= 1 recommended screenshot timestamp.
- Refer to video times in MM:SS format (e.g. 01:23).
- Output MUST be a single valid JSON object matching the schema exactly. \
  Do NOT wrap it in markdown code fences or add any text outside the JSON.
"""

# ═══════════════════════════════════════════════════════════════════════
#  User prompt
# ═══════════════════════════════════════════════════════════════════════

_SOP_SCHEMA_TEMPLATE = """\
Analyze the screen recording above and produce a structured SOP.

{title_line}
{domain_line}

Return a JSON object with EXACTLY this schema (no extra keys):

{{
  "title": "<string>",
  "intro": "<short paragraph>",
  "settings": {{
    "max_substeps_per_step": 4,
    "min_images_per_step": 1
  }},
  "steps": [
    {{
      "step_number": <int starting at 1>,
      "step_title": "<concise title>",
      "substeps": [
        "<imperative sentence — max 4 items>"
      ],
      "evidence": {{
        "recommended_screenshot_timestamps": ["MM:SS"],
        "supporting_timestamps": [
          {{"start": "MM:SS", "end": "MM:SS", "why": "<reason>"}}
        ]
      }},
      "images": [
        {{"image_id": "step_<N>_img_<M>", "caption": "<what screenshot shows>"}}
      ]
    }}
  ],
  "warnings": ["<optional caveat strings>"]
}}

Constraints:
- 1–4 substeps per step (imperative voice).
- >= 1 recommended_screenshot_timestamps per step.
- image_id format: step_<step_number>_img_<1-based index>.
- images[].caption must describe what the screenshot shows for that step.
- Timestamps must be in MM:SS format referencing real moments in the video.
- Output ONLY valid JSON. No markdown, no commentary.
"""


def build_sop_prompt(
    title_hint: Optional[str] = None,
    domain_hint: Optional[str] = None,
) -> str:
    """Build the user-content text prompt placed after the video part."""
    title_line = (
        f'Use this as the SOP title: "{title_hint}"'
        if title_hint
        else "Generate an appropriate SOP title from the video content."
    )
    domain_line = (
        f'Domain context: {domain_hint}'
        if domain_hint
        else ""
    )
    return _SOP_SCHEMA_TEMPLATE.format(
        title_line=title_line,
        domain_line=domain_line,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Repair prompt (text-only, no video)
# ═══════════════════════════════════════════════════════════════════════

_REPAIR_TEMPLATE = """\
The SOP JSON you produced has validation errors. Fix them and return the \
corrected JSON.

VALIDATION ERRORS:
{error_list}

ORIGINAL JSON:
{original_json}

Rules reminder:
- Maximum 4 substeps per step.
- Minimum 1 recommended_screenshot_timestamps per step.
- Minimum 1 images entry per step.
- image_id format: step_<N>_img_<M>.
- Timestamps in MM:SS.
- Output ONLY valid JSON — no markdown, no explanation.
"""


def build_repair_prompt(original_json: str, errors: list[str]) -> str:
    error_list = "\n".join(f"  - {e}" for e in errors)
    return _REPAIR_TEMPLATE.format(
        error_list=error_list,
        original_json=original_json,
    )
