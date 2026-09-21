"""Pydantic request/response schemas for the HTTP API.

Kept separate from app/classifier.py on purpose: the classifier module
speaks in domain terms (MatchResult) and knows nothing about HTTP, while
this module owns the wire format and its validation rules.
"""

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    resume: str = Field(..., min_length=1, description="Full resume text.")
    job_description: str = Field(
        ..., min_length=1, description="Full job description text."
    )


class ResumeInput(BaseModel):
    id: str | None = Field(
        default=None,
        description="Caller-supplied identifier, echoed back in the result. "
        "Defaults to 'resume-<index>' if omitted.",
    )
    resume: str = Field(..., min_length=1, description="Full resume text.")


class BatchMatchRequest(BaseModel):
    resumes: list[ResumeInput] = Field(..., min_length=1, max_length=50)
    job_description: str = Field(
        ..., min_length=1, description="Full job description text."
    )


class MatchResponse(BaseModel):
    fit_score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized overall fit, 0-1."
    )
    fit_confidence: float = Field(..., ge=0.0, le=1.0)
    fit_legend: str = Field(
        ..., description="The criteria description Jev's fit_score is closest to."
    )
    has_required_skills: bool
    has_required_skills_probability: float = Field(..., ge=0.0, le=1.0)
    seniority_match: str
    seniority_confidence: float = Field(..., ge=0.0, le=1.0)
    model: str = Field(..., description="Versioned Jev model ID that answered.")
    needs_human_review: bool = Field(
        ..., description="True if any relevant confidence fell below the gate."
    )


class BatchMatchResultItem(MatchResponse):
    resume_id: str


class BatchMatchResponse(BaseModel):
    results: list[BatchMatchResultItem] = Field(
        ..., description="Sorted by fit_score, descending."
    )


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
