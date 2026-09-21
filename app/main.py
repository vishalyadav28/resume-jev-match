"""FastAPI service for resume-to-job-description matching via Jev.

Two endpoints:
  POST /match        one resume vs one job description
  POST /batch-match   many resumes vs one job description, via .batch()
  GET  /health        liveness + whether TYPESAFE_API_KEY is configured
"""

import logging

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_typesafe.client import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafeRateLimitError,
)

from app.classifier import MatchResult, match_resume, match_resumes
from app.config import Settings, get_settings
from app.models import (
    BatchMatchRequest,
    BatchMatchResponse,
    BatchMatchResultItem,
    ErrorDetail,
    ErrorResponse,
    MatchRequest,
    MatchResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resume_jev_match")

app = FastAPI(
    title="resume-jev-match",
    description=(
        "Resume-to-job-description fit classification, powered by "
        "TypeSafe AI's Jev model."
    ),
    version="0.1.0",
)


def _require_configured(settings: Settings) -> None:
    """Fail fast, with a clean error, when no API key is set at all.

    This is distinct from an *invalid* key, which TypeSafe itself rejects
    and which we surface via the TypeSafeAPIError handler below.
    """
    if not settings.typesafe_api_key:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "missing_api_key",
                    "message": "TYPESAFE_API_KEY is not set on the server.",
                }
            },
        )


def _needs_human_review(result: MatchResult, threshold: float) -> bool:
    """Gate on the two confidence-bearing answers.

    Noul (has_required_skills) has no confidence field by design, so only
    fit_confidence and seniority_confidence can drive this gate.
    """
    return min(result.fit_confidence, result.seniority_confidence) < threshold


def _to_response(result: MatchResult, threshold: float) -> MatchResponse:
    return MatchResponse(
        fit_score=result.fit_score,
        fit_confidence=result.fit_confidence,
        fit_legend=result.fit_legend,
        has_required_skills=result.has_required_skills,
        has_required_skills_probability=result.has_required_skills_probability,
        seniority_match=result.seniority_match,
        seniority_confidence=result.seniority_confidence,
        model=result.model,
        needs_human_review=_needs_human_review(result, threshold),
    )


@app.exception_handler(TypeSafeAuthenticationError)
async def handle_authentication_error(
    request: Request, exc: TypeSafeAuthenticationError
) -> JSONResponse:
    logger.error("TypeSafe rejected the API key: %s", exc)
    return JSONResponse(
        status_code=401,
        content=ErrorResponse(
            error=ErrorDetail(
                code="authentication_failed",
                message="TypeSafe rejected the API key. Check TYPESAFE_API_KEY.",
            )
        ).model_dump(),
    )


@app.exception_handler(TypeSafeRateLimitError)
async def handle_rate_limit(
    request: Request, exc: TypeSafeRateLimitError
) -> JSONResponse:
    logger.warning("Jev rate limit hit: request_id=%s", exc.request_id)
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            error=ErrorDetail(
                code="rate_limited",
                message="Jev is rate-limiting requests right now. Retry shortly.",
            )
        ).model_dump(),
    )


@app.exception_handler(TypeSafeAPIConnectionError)
async def handle_connection_error(
    request: Request, exc: TypeSafeAPIConnectionError
) -> JSONResponse:
    # Covers both connection failures and timeouts (TypeSafeAPITimeoutError
    # subclasses this) -- neither carries an HTTP status, since no response
    # was ever received.
    logger.error("Could not reach TypeSafe: %s", exc)
    return JSONResponse(
        status_code=502,
        content=ErrorResponse(
            error=ErrorDetail(
                code="upstream_unreachable",
                message="Could not reach the Jev API.",
            )
        ).model_dump(),
    )


@app.exception_handler(TypeSafeAPIError)
async def handle_typesafe_api_error(
    request: Request, exc: TypeSafeAPIError
) -> JSONResponse:
    # Catch-all for the remaining TypeSafeAPIError subclasses (bad request,
    # permission denied, not found, unprocessable entity, internal server
    # error) that don't need their own dedicated handler above.
    logger.error("TypeSafe API error: status=%s detail=%s", exc.status, exc)
    return JSONResponse(
        status_code=exc.status,
        content=ErrorResponse(
            error=ErrorDetail(
                code="typesafe_api_error", message="Jev API request failed."
            )
        ).model_dump(),
    )


@app.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "typesafe_configured": settings.typesafe_api_key is not None,
    }


@app.post("/match", response_model=MatchResponse)
def match(
    payload: MatchRequest, settings: Settings = Depends(get_settings)
) -> MatchResponse:
    _require_configured(settings)
    result = match_resume(payload.resume, payload.job_description)
    logger.info("match: model=%s fit_score=%.3f", result.model, result.fit_score)
    return _to_response(result, settings.confidence_gate_threshold)


@app.post("/batch-match", response_model=BatchMatchResponse)
def batch_match(
    payload: BatchMatchRequest, settings: Settings = Depends(get_settings)
) -> BatchMatchResponse:
    _require_configured(settings)
    resume_ids = [
        resume.id or f"resume-{index}" for index, resume in enumerate(payload.resumes)
    ]
    results = match_resumes(
        [resume.resume for resume in payload.resumes], payload.job_description
    )
    items = [
        BatchMatchResultItem(
            resume_id=resume_id,
            **_to_response(result, settings.confidence_gate_threshold).model_dump(),
        )
        for resume_id, result in zip(resume_ids, results, strict=True)
    ]
    items.sort(key=lambda item: item.fit_score, reverse=True)
    logger.info("batch_match: n=%d", len(items))
    return BatchMatchResponse(results=items)
