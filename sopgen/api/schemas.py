"""Pydantic models for the SOP domain objects and API request/response contracts."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════
#  SOP domain models  (matches the strict JSON schema Gemini must emit)
# ═══════════════════════════════════════════════════════════════════════


class SOPSettings(BaseModel):
    max_substeps_per_step: int = 4
    min_images_per_step: int = 1


class SupportingTimestamp(BaseModel):
    start: str = Field(..., description="MM:SS")
    end: str = Field(..., description="MM:SS")
    why: str


class StepEvidence(BaseModel):
    recommended_screenshot_timestamps: List[str] = Field(
        ..., min_length=1, description="At least one MM:SS timestamp"
    )
    supporting_timestamps: List[SupportingTimestamp] = Field(default_factory=list)


class StepImage(BaseModel):
    image_id: str
    caption: str


class SOPStep(BaseModel):
    step_number: int
    step_title: str = Field(..., min_length=1)
    substeps: List[str] = Field(
        ..., min_length=1, max_length=4, description="1–4 imperative bullets"
    )
    evidence: StepEvidence
    images: List[StepImage] = Field(default_factory=list)

    @field_validator("substeps")
    @classmethod
    def cap_substeps(cls, v: list[str]) -> list[str]:
        if len(v) > 4:
            raise ValueError(
                f"Maximum 4 substeps per step, got {len(v)}"
            )
        return v


class SOPDocument(BaseModel):
    title: str = Field(..., min_length=1)
    intro: str = Field(..., min_length=1)
    settings: SOPSettings = Field(default_factory=SOPSettings)
    steps: List[SOPStep] = Field(..., min_length=1)
    warnings: List[str] = Field(default_factory=list)

    @field_validator("steps")
    @classmethod
    def check_image_requirement(cls, v: list[SOPStep]) -> list[SOPStep]:
        for i, step in enumerate(v):
            ts = step.evidence.recommended_screenshot_timestamps
            if len(ts) < 1:
                raise ValueError(
                    f"Step {i + 1} must have >= 1 recommended screenshot timestamp"
                )
        return v


# ═══════════════════════════════════════════════════════════════════════
#  API request / response models
# ═══════════════════════════════════════════════════════════════════════


class ImageOut(BaseModel):
    image_id: str
    url: str
    caption: str


class SOPResponse(BaseModel):
    job_id: str
    sop: SOPDocument
    image_base_url: str
    images: List[ImageOut] = Field(default_factory=list)


class SOPErrorResponse(BaseModel):
    detail: str
    supported_types: Optional[List[str]] = None
    retryable: bool = False
