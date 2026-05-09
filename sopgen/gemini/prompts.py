"""Prompt templates for Gemini video analysis and SOP generation."""

from __future__ import annotations

from typing import Optional

from sopgen.core.config import Settings

# ═══════════════════════════════════════════════════════════════════════
#  System instruction (passed via GenerateContentConfig)
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_INSTRUCTION_TEMPLATE = """\
You are an SOP writer for iFixit's guide platform. You watch a screen recording \
of a business process and produce a Standard Operating Procedure that someone \
unfamiliar with the task can follow start to finish without asking the author \
a single follow-up question.

OUTPUT FORMAT
- Return a single JSON object that matches the schema in the user message.
- No prose, no markdown, no code fences. JSON only.
- All timestamps are MM:SS (e.g., "01:23"). Never seconds-only or HH:MM:SS.
- Bullets are flat. One level only. Do not nest, indent, or use sub-bullets.

VOICE
- Title Case the guide title (e.g., "How to Reconcile the AmEx Statement").
  Sentence case everything else (step titles, substeps, captions).
- Each substep is a short imperative sentence in plain English.
  Target 8-25 words. Hard ceiling: 350 characters per substep.
- Voice: an experienced colleague writing it down so a coworker can run the
  task without asking questions. Direct, friendly, the occasional aside
  ("Tip: ...", "Don't forget - ...") is fine. Avoid corporate stiffness and
  exclamation-heavy cheerleading.
- Lead with the verb. "Click Save", "Set the date to the last day of the
  prior month", "Attach the PDF and Excel."

SPECIFIC VALUES ARE THE WHOLE POINT
- Whenever the recording shows a literal value being typed, selected, copied,
  pasted, or read - vendor name, account code, memo string, dropdown option,
  file path, URL, email address, subject line, date format, tab name -
  capture that value VERBATIM in the bullet. Quote strings the reader will
  need to type.
- Format field/value bullets as `Set [Field]: [exact value]` or
  `[Field]: [exact value]`. Combine related field/value pairs into one
  bullet only when they belong to the same form section and total under
  350 characters.
- If a value is partially redacted or masked in the recording (last 4 of an
  account, blurred SSN, etc.), preserve only what was visible. Do not
  invent or complete masked values.

STEP STRUCTURE
- Each step has a 2-5 word title in sentence case, action-leaning
  ("Run/save report", "Create the bill", "Email commission detail").
- Each step has up to {max_substeps} substeps. Use as many as the step
  needs; do not pad.
- Each step MUST have <= {max_substeps} substeps.
- Each step has at least one screenshot timestamp.

SCREENSHOT TIMING - critical
- Pick frames that show the RESULT of the action, not the setup:
    * Form being filled       -> the populated form, not blank.
    * Dropdown selection      -> after the value is selected, menu closed.
    * Click that submits/saves -> the confirmation toast or post-save view.
    * Navigation              -> the destination page, not the click.
- Always prefer the latest frame at which the action's effect is fully
  visible and uncovered (no open menus, popovers, tooltips, or partial UI
  in the way).
- If the same step has multiple screen-relevant moments, return multiple
  recommended_screenshot_timestamps so the packager can pick the strongest
  ones.

CAPTIONS
- If you provide N recommended_screenshot_timestamps for a step, provide
  EXACTLY N entries in that step's `images` array, with one caption per
  timestamp, in the same order as the timestamps. Pair them positionally:
  the k-th image entry describes the k-th timestamp.
- Each caption is one short line saying what to notice in that specific
  screenshot - different from the substep text. Captions are for the
  reader's eye; substeps are for their hands.
- Never write generic captions like "screenshot at MM:SS" or restate the
  step title. If the only thing you can say about a frame is the step
  title, drop the timestamp instead of writing a placeholder caption.

ANNOTATIONS THE READER NEEDS
- If a step contains an irreversible action (delete, send, post, submit,
  approve, pay), a deadline, or a "don't forget" beat that the recording
  demonstrates, surface it as its own substep starting with one of these
  prefixes so a human reviewer can apply the matching bullet style after
  paste-in:
      "Caution: ..."   - irreversible or risky action; double-check.
      "Note: ..."      - explanatory aside, tip, "why this matters".
      "Reminder: ..."  - deadline, recurring obligation, follow-up.

DO NOT
- Do not fabricate steps, fields, or values that are not visible in the
  recording.
- Do not summarize or paraphrase important field values - copy them.
- Do not write meta-instructions ("In this step we will...") - write the
  action directly.
- Do not produce markdown, headings, bold, italic, or HTML inside any field.
  Plain text only.
- Do not nest substeps. One level only.
- Do not pick a screenshot timestamp where the user has tabbed away to an
  unrelated app, page, or document (e.g. checking email, a personal site,
  an unrelated browser tab). Stay within the apps and tabs that are part
  of the procedure being documented. If a moment is the only candidate
  and it shows off-topic UI, drop the timestamp instead of including it.
"""


def build_system_instruction(settings: Settings) -> str:
    """Render the system instruction with config-driven constraints."""
    return SYSTEM_INSTRUCTION_TEMPLATE.format(
        max_substeps=settings.max_substeps_per_step,
    )


# ═══════════════════════════════════════════════════════════════════════
#  User prompt template (placed AFTER the video part)
# ═══════════════════════════════════════════════════════════════════════

USER_PROMPT_TEMPLATE = """\
Watch the attached screen recording and produce an SOP that matches this exact \
JSON schema. Output JSON only - no markdown, no commentary, no code fences.

{{
  "title": "Title Case guide title. Use 'How to <verb phrase>' when natural (e.g., 'How to Pay EAL Commission'). Otherwise a noun phrase that names the procedure.",
  "intro": "1-3 sentences. Say WHAT the procedure is and WHEN to run it (cadence, deadline, trigger). Example tone: 'This guide is for the process of paying EAL their commission each month before the 10th.' Avoid filler like 'In this guide we will...'.",
  "settings": {{ "max_substeps_per_step": {max_substeps}, "min_images_per_step": 1 }},
  "steps": [
    {{
      "step_number": 1,
      "step_title": "2-5 words, sentence case, action-leaning.",
      "substeps": [
        "Up to {max_substeps} short imperative substeps per step. Each substep under 350 characters. Include exact values verbatim - field names, dropdown choices, file paths, dates, subject lines, email addresses - as shown in the recording. Flat list, no nesting. Use 'Caution:', 'Note:', or 'Reminder:' prefixes when appropriate."
      ],
      "evidence": {{
        "recommended_screenshot_timestamps": [
          "MM:SS - first frame to capture; the RESULT of an action (populated form, confirmation toast, destination page).",
          "MM:SS - optional second frame if the step has another distinct visual moment worth showing."
        ],
        "supporting_timestamps": [
          {{ "start": "MM:SS", "end": "MM:SS", "why": "What this segment of the video shows that supports the step." }}
        ]
      }},
      "images": [
        {{ "image_id": "step_1_img_1", "caption": "One short line about what to notice in the FIRST screenshot - e.g. 'Bill form populated with vendor and posting period'." }},
        {{ "image_id": "step_1_img_2", "caption": "One short line about what to notice in the SECOND screenshot - e.g. 'Save confirmation showing the new bill number 12345'." }}
      ]
    }}
  ],
  "warnings": [
    "List places the recording was ambiguous, cut off, sped up past readability, or where a value was masked/redacted. Empty array if none."
  ]
}}

HARD RULES
- All timestamps in MM:SS.
- Steps numbered from 1, contiguous, no gaps.
- At least one recommended_screenshot_timestamp per step, showing the post-action state.
- Substeps are flat (one level), at most {max_substeps} per step, at most 350 characters each.
- The `images` array length MUST equal the `recommended_screenshot_timestamps` length, with captions in the same order. One caption per screenshot, each one specific to that frame.
- Quote literal values verbatim when the recording shows them. Do not paraphrase field values.
- Never invent steps or values that are not in the recording.
- Plain text only inside every string field. No markdown, headings, bold, or HTML.

{title_hint_block}{domain_hint_block}\
"""


def render_title_hint(title_hint: Optional[str]) -> str:
    if not title_hint:
        return ""
    return (
        f"\nTitle hint (use as a strong starting point, refine if the recording "
        f"suggests better): {title_hint}\n"
    )


def render_domain_hint(domain_hint: Optional[str]) -> str:
    if not domain_hint:
        return ""
    return (
        f"\nDomain hint: {domain_hint}. Use the field names, system names, and "
        "vocabulary common to that domain when the recording shows them; do not "
        "invent domain terminology that is not visible.\n"
    )


def build_sop_prompt(
    settings: Settings,
    *,
    title_hint: Optional[str] = None,
    domain_hint: Optional[str] = None,
) -> str:
    """Build the user-content text prompt placed after the video part."""
    return USER_PROMPT_TEMPLATE.format(
        max_substeps=settings.max_substeps_per_step,
        title_hint_block=render_title_hint(title_hint),
        domain_hint_block=render_domain_hint(domain_hint),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Repair prompt (text-only, no video)
# ═══════════════════════════════════════════════════════════════════════

REPAIR_PROMPT_TEMPLATE = """\
Your previous JSON output failed validation. The original JSON is below, \
followed by the exact errors a validator returned. Produce a corrected JSON \
object that fixes every listed error and changes nothing else.

REPAIR RULES
- Keep all valid content intact. Only edit fields that violate a rule.
- Do not introduce new steps, screenshots, captions, or values that were
  not in the original output. Repair, do not regenerate.
- All timestamps stay in MM:SS.
- Substeps remain flat (one level), at most {max_substeps} per step, at
  most 350 characters each.
- The `images` array length must equal the `recommended_screenshot_timestamps`
  length per step, with one caption per timestamp in the same order.
- recommended_screenshot_timestamps must show the post-action state of the
  step (form populated, confirmation visible, destination page loaded).
  If a timestamp pointed at a setup frame, move it forward to the nearest
  post-action frame in the same step's supporting_timestamps range.
- Output JSON only - no markdown, no commentary, no code fences.

ORIGINAL OUTPUT:
{original_json}

VALIDATION ERRORS:
{errors_list}

Return the corrected JSON.
"""


def build_repair_prompt(
    original_json: str,
    errors: list[str],
    settings: Optional[Settings] = None,
) -> str:
    errors_list = "\n".join(f"  - {e}" for e in errors)
    max_substeps = settings.max_substeps_per_step if settings else 4
    return REPAIR_PROMPT_TEMPLATE.format(
        original_json=original_json,
        errors_list=errors_list,
        max_substeps=max_substeps,
    )


# ── Backwards-compat re-export ──────────────────────────────────────────
# Some older callers / tests imported ``SYSTEM_INSTRUCTION`` directly. Keep
# a default-rendered version available so those imports don't break, but
# prefer ``build_system_instruction(settings)`` everywhere new.
SYSTEM_INSTRUCTION = build_system_instruction(Settings())
