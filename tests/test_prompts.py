"""Tests for prompt-rendering invariants.

These cover the dynamic injection of ``settings.max_substeps_per_step``
into both the system instruction and the user prompt, plus the new
CAPTIONS and off-topic-frame guardrail sections.
"""

from __future__ import annotations

import re

import pytest

from sopgen.core.config import Settings
from sopgen.gemini.prompts import (
    build_repair_prompt,
    build_sop_prompt,
    build_system_instruction,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in (
        "SOPGEN_MAX_SUBSTEPS_PER_STEP",
        "SOPGEN_GEMINI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


# ── Dynamic max_substeps ────────────────────────────────────────────────


class TestDynamicMaxSubsteps:
    @pytest.mark.parametrize("n", [3, 4, 5, 7, 10])
    def test_both_prompts_use_same_n(self, n):
        settings = Settings(max_substeps_per_step=n)
        sys_text = build_system_instruction(settings)
        usr_text = build_sop_prompt(settings)

        # System instruction: "up to N substeps", "<= N substeps"
        assert f"up to {n} substeps" in sys_text
        assert f"<= {n} substeps" in sys_text

        # User prompt: schema example, substeps description, hard rule
        assert f'"max_substeps_per_step": {n}' in usr_text
        assert f"Up to {n} short imperative substeps" in usr_text
        assert f"at most {n} per step" in usr_text

    def test_no_stray_literal_4_when_config_is_seven(self):
        settings = Settings(max_substeps_per_step=7)
        sys_text = build_system_instruction(settings)
        usr_text = build_sop_prompt(settings)

        stray_patterns = [
            r"up to 4(?!\d)",
            r"<= 4(?!\d)",
            r"at most 4(?!\d)",
            r'max_substeps_per_step":\s*4(?!\d)',
        ]
        for txt, label in [(sys_text, "system"), (usr_text, "user")]:
            for pat in stray_patterns:
                assert not re.search(pat, txt), (
                    f"stray substep cap '4' found in {label} prompt: {pat}"
                )

    def test_repair_prompt_uses_dynamic_when_settings_passed(self):
        prompt = build_repair_prompt(
            "{}",
            ["err"],
            settings=Settings(max_substeps_per_step=6),
        )
        assert "at most 6 per step" in prompt

    def test_repair_prompt_defaults_to_four_without_settings(self):
        prompt = build_repair_prompt("{}", ["err"])
        assert "at most 4 per step" in prompt


# ── CAPTIONS section ────────────────────────────────────────────────────


class TestCaptionsSection:
    def test_system_instruction_has_captions_section(self):
        text = build_system_instruction(Settings())
        assert "CAPTIONS" in text
        # Core rule: N timestamps -> N image entries, in order
        assert "EXACTLY N entries" in text
        assert "same order as the timestamps" in text
        # Anti-pattern: no generic placeholders
        assert "screenshot at MM:SS" in text  # mentioned as a thing not to do

    def test_user_prompt_example_shows_two_image_entries(self):
        text = build_sop_prompt(Settings())
        # The image example must show TWO entries with distinct captions
        # so the model copies the pattern.
        assert text.count('"image_id": "step_1_img_1"') == 1
        assert text.count('"image_id": "step_1_img_2"') == 1
        assert "FIRST screenshot" in text
        assert "SECOND screenshot" in text


# ── Off-topic frame guardrail ───────────────────────────────────────────


class TestOffTopicGuardrail:
    def test_do_not_pick_off_topic_frames(self):
        text = build_system_instruction(Settings())
        assert "Do not pick a screenshot timestamp where the user has tabbed away" in text
        assert "unrelated app" in text or "unrelated browser tab" in text


# ── Hint blocks still work ──────────────────────────────────────────────


class TestHintBlocks:
    def test_title_and_domain_hints_appear(self):
        text = build_sop_prompt(
            Settings(),
            title_hint="How to Pay EAL Commission",
            domain_hint="NetSuite AP",
        )
        assert "How to Pay EAL Commission" in text
        assert "NetSuite AP" in text

    def test_no_hint_block_when_omitted(self):
        text = build_sop_prompt(Settings())
        assert "Title hint" not in text
        assert "Domain hint" not in text
