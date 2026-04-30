"""Schema validation and auto-repair loop for Gemini SOP output."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from pydantic import ValidationError

from sopgen.api.schemas import SOPDocument

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    sop: Optional[SOPDocument] = None
    errors: list[str] = field(default_factory=list)


class SOPValidator:
    """Validates raw JSON against the SOPDocument schema."""

    def validate(self, raw_json: str) -> ValidationResult:
        """Parse and validate a JSON string.

        Returns a ``ValidationResult`` with ``is_valid=True`` and the
        parsed ``SOPDocument`` on success, or ``is_valid=False`` with a
        list of human-readable error strings.
        """
        # 1) JSON parse
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            return ValidationResult(
                is_valid=False,
                errors=[f"Invalid JSON: {exc}"],
            )

        # 2) Pydantic schema validation
        try:
            sop = SOPDocument.model_validate(data)
        except ValidationError as exc:
            errors = [
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            ]
            return ValidationResult(is_valid=False, errors=errors)

        # 3) Extra business-rule checks that Pydantic can't express easily
        extra_errors: list[str] = []
        for i, step in enumerate(sop.steps):
            if len(step.substeps) > 4:
                extra_errors.append(
                    f"Step {i + 1}: has {len(step.substeps)} substeps (max 4)"
                )
            if len(step.evidence.recommended_screenshot_timestamps) < 1:
                extra_errors.append(
                    f"Step {i + 1}: missing recommended screenshot timestamps"
                )

        if extra_errors:
            return ValidationResult(is_valid=False, errors=extra_errors)

        return ValidationResult(is_valid=True, sop=sop)


def run_with_repair(
    analyzer: object,
    video_path: object,
    mime_type: str,
    *,
    title_hint: Optional[str] = None,
    domain_hint: Optional[str] = None,
    fps_override: Optional[int] = None,
    max_retries: int = 2,
) -> SOPDocument:
    """End-to-end: analyze → validate → repair loop → return SOPDocument.

    Parameters
    ----------
    analyzer : VideoAnalyzer
        Instance with ``analyze()`` and ``repair()`` methods.
    video_path, mime_type, title_hint, domain_hint, fps_override :
        Forwarded to ``analyzer.analyze()``.
    max_retries :
        Number of repair attempts before raising.

    Raises
    ------
    ValueError
        If the SOP cannot be made valid within *max_retries* passes.
    """
    from sopgen.gemini.video_analyze import VideoAnalyzer  # local import to avoid cycle

    assert isinstance(analyzer, VideoAnalyzer)

    validator = SOPValidator()

    # Initial generation
    raw_json = analyzer.analyze(
        video_path,  # type: ignore[arg-type]
        mime_type,
        title_hint=title_hint,
        domain_hint=domain_hint,
        fps_override=fps_override,
    )
    result = validator.validate(raw_json)

    # Repair loop
    attempts = 0
    while not result.is_valid and attempts < max_retries:
        attempts += 1
        logger.warning(
            "Validation failed (attempt %d/%d): %s",
            attempts,
            max_retries,
            result.errors,
        )
        raw_json = analyzer.repair(raw_json, result.errors)
        result = validator.validate(raw_json)

    if not result.is_valid:
        raise ValueError(
            f"SOP validation failed after {attempts} repair attempts. "
            f"Errors: {result.errors}"
        )

    logger.info("SOP validated successfully (repair_attempts=%d)", attempts)
    assert result.sop is not None
    return result.sop
