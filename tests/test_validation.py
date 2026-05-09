"""Tests for SOP validation and repair prompt generation."""

import json

import pytest

from sopgen.core.validation import SOPValidator
from sopgen.gemini.prompts import build_repair_prompt


def _make_sop(**overrides) -> dict:
    """Minimal valid SOP dict; apply *overrides* on top."""
    base = {
        "title": "Test SOP",
        "intro": "This procedure covers a test workflow.",
        "settings": {"max_substeps_per_step": 4, "min_images_per_step": 1},
        "steps": [
            {
                "step_number": 1,
                "step_title": "Open the application",
                "substeps": ["Navigate to the home screen"],
                "evidence": {
                    "recommended_screenshot_timestamps": ["00:05"],
                    "supporting_timestamps": [],
                },
                "images": [
                    {"image_id": "step_1_img_1", "caption": "Home screen"}
                ],
            }
        ],
        "warnings": [],
    }
    base.update(overrides)
    return base


# ── Happy path ──────────────────────────────────────────────────────────


class TestValidSOP:
    def test_minimal_valid(self):
        validator = SOPValidator()
        result = validator.validate(json.dumps(_make_sop()))
        assert result.is_valid
        assert result.sop is not None
        assert result.sop.title == "Test SOP"

    def test_multiple_steps(self):
        sop = _make_sop()
        sop["steps"].append(
            {
                "step_number": 2,
                "step_title": "Submit form",
                "substeps": ["Click Save", "Confirm dialog"],
                "evidence": {
                    "recommended_screenshot_timestamps": ["01:30"],
                    "supporting_timestamps": [
                        {"start": "01:25", "end": "01:35", "why": "form visible"}
                    ],
                },
                "images": [
                    {"image_id": "step_2_img_1", "caption": "Save button"}
                ],
            }
        )
        result = SOPValidator().validate(json.dumps(sop))
        assert result.is_valid
        assert len(result.sop.steps) == 2


# ── Constraint violations ───────────────────────────────────────────────


class TestSubstepLimit:
    """The substep cap is parameterized — these tests verify the upper
    bound is enforced regardless of the configured value."""

    @pytest.mark.parametrize("cap", [4, 5, 7, 10])
    def test_one_more_than_cap_rejected(self, cap):
        sop = _make_sop()
        sop["steps"][0]["substeps"] = [f"item {i}" for i in range(cap + 1)]
        result = SOPValidator(max_substeps=cap).validate(json.dumps(sop))
        assert not result.is_valid
        # Error message should mention the configured cap, not a hardcoded 4.
        assert any(f"max {cap}" in e for e in result.errors)

    @pytest.mark.parametrize("cap", [4, 5, 7, 10])
    def test_exactly_cap_accepted(self, cap):
        sop = _make_sop()
        sop["steps"][0]["substeps"] = [f"item {i}" for i in range(cap)]
        result = SOPValidator(max_substeps=cap).validate(json.dumps(sop))
        assert result.is_valid, result.errors

    def test_default_cap_is_four(self):
        """Existing callers that pass no args still get the historical cap=4."""
        sop = _make_sop()
        sop["steps"][0]["substeps"] = ["a", "b", "c", "d", "e"]  # 5 > 4
        result = SOPValidator().validate(json.dumps(sop))
        assert not result.is_valid
        assert any("max 4" in e for e in result.errors)

    def test_seven_substeps_pass_when_cap_is_seven(self):
        """The original motivating case: SOPGEN_MAX_SUBSTEPS_PER_STEP=7."""
        sop = _make_sop()
        sop["steps"][0]["substeps"] = [f"step {i}" for i in range(7)]
        result = SOPValidator(max_substeps=7).validate(json.dumps(sop))
        assert result.is_valid, result.errors

    def test_eight_substeps_fail_when_cap_is_seven(self):
        sop = _make_sop()
        sop["steps"][0]["substeps"] = [f"step {i}" for i in range(8)]
        result = SOPValidator(max_substeps=7).validate(json.dumps(sop))
        assert not result.is_valid

    def test_zero_substeps_always_rejected(self):
        """min_length=1 is structural and stays regardless of cap."""
        sop = _make_sop()
        sop["steps"][0]["substeps"] = []
        result = SOPValidator(max_substeps=10).validate(json.dumps(sop))
        assert not result.is_valid


class TestMissingScreenshotTimestamps:
    def test_empty_timestamps_rejected(self):
        sop = _make_sop()
        sop["steps"][0]["evidence"]["recommended_screenshot_timestamps"] = []
        result = SOPValidator().validate(json.dumps(sop))
        assert not result.is_valid

    def test_no_evidence_key_rejected(self):
        sop = _make_sop()
        del sop["steps"][0]["evidence"]
        result = SOPValidator().validate(json.dumps(sop))
        assert not result.is_valid


class TestBadJSON:
    def test_invalid_json(self):
        result = SOPValidator().validate("{not json at all")
        assert not result.is_valid
        assert any("json" in e.lower() for e in result.errors)

    def test_empty_string(self):
        result = SOPValidator().validate("")
        assert not result.is_valid


# ── Repair prompt ───────────────────────────────────────────────────────


class TestRepairPrompt:
    def test_includes_errors(self):
        prompt = build_repair_prompt('{"bad": true}', ["Missing title", "No steps"])
        assert "Missing title" in prompt
        assert "No steps" in prompt
        assert '{"bad": true}' in prompt

    def test_asks_for_json(self):
        prompt = build_repair_prompt("{}", ["err"])
        assert "JSON" in prompt
