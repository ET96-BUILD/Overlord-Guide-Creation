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
    def test_five_substeps_rejected(self):
        sop = _make_sop()
        sop["steps"][0]["substeps"] = [
            "One", "Two", "Three", "Four", "Five"
        ]
        result = SOPValidator().validate(json.dumps(sop))
        assert not result.is_valid
        assert any("substep" in e.lower() or "4" in e for e in result.errors)


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
