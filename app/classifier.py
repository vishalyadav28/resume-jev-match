"""Resume-to-job-description matching, powered by TypeSafe AI's Jev model.

Jev is a "System One" model: instead of generating text that then has to be
parsed, it takes a piece of state plus a set of typed questions and returns
typed, calibrated answers directly (a probability, a label, or a position on
a scale). ``TypeSafeClassifier`` exposes that as a LangChain ``Runnable``, so
this module never touches raw JSON or does any output parsing -- it just
declares questions and reads typed fields off the response.

This module has no FastAPI dependency by design: it is reusable and testable
on its own, independent of how it is served.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier
from langchain_typesafe.types import ClassifierRequest, ClassifierResponse
from pydantic import JsonValue

from app.config import get_settings

# Score criteria are ordered low -> high. Jev's `.score` is a
# probability-weighted mean of the level *index* (0, 1, 2, ...), not a 0-1
# probability, so we normalize by the number of steps to get a 0-1 fit score
# in the API response.
_OVERALL_FIT_CRITERIA: list[JsonValue] = [
    "No meaningful overlap in skills or experience",
    "Some relevant skills, but key requirements are missing",
    "Strong match — most required skills and experience present",
]

_SENIORITY_CRITERIA: dict[str, JsonValue] = {
    "underqualified": (
        "Resume shows meaningfully less experience or seniority than the "
        "role requires."
    ),
    "matched": (
        "Resume's experience and seniority line up with what the role requires."
    ),
    "overqualified": (
        "Resume shows meaningfully more experience or seniority than the "
        "role requires."
    ),
    "unclear": (
        "Resume or job description doesn't give enough information to "
        "judge seniority fit."
    ),
}


@dataclass(frozen=True)
class MatchResult:
    """Typed result of scoring one resume against one job description."""

    fit_score: float  # normalized 0-1, higher = better fit
    fit_confidence: float
    fit_legend: str  # the criteria description closest to fit_score
    has_required_skills: bool
    has_required_skills_probability: float  # raw Noul probability, 0-1
    seniority_match: str  # one of _SENIORITY_CRITERIA's keys
    seniority_confidence: float
    model: str  # versioned Jev model ID that answered, e.g. "jev-2026-09-15"
    request_id: str | None


def _build_state(resume_text: str, job_description_text: str) -> str:
    """Build the minimal state Jev needs to judge fit.

    Accuracy drops when irrelevant content is mixed into the state, so we
    pass only the resume and JD text, trimmed, with clear section labels --
    nothing else (no filenames, timestamps, or metadata the questions below
    don't need).
    """
    return (
        f"RESUME:\n{resume_text.strip()}\n\n"
        f"JOB DESCRIPTION:\n{job_description_text.strip()}"
    )


def _build_request(resume_text: str, job_description_text: str) -> ClassifierRequest:
    return {
        "state": _build_state(resume_text, job_description_text),
        "questions": _build_questions(),
    }


@lru_cache(maxsize=1)
def get_classifier() -> TypeSafeClassifier:
    """Lazily construct a shared TypeSafeClassifier.

    Lazy + cached so importing this module never requires TYPESAFE_API_KEY
    to be set (useful for tests that don't hit the network), while real
    requests still reuse one classifier instance instead of building a new
    one per call.

    Reads api_key from our own Settings (which loads .env) rather than
    letting TypeSafeClassifier() read the raw process environment --
    pydantic-settings loading a .env file does not export it into
    os.environ, so the two would otherwise disagree about whether a key is
    configured.
    """
    settings = get_settings()
    if settings.typesafe_api_key:
        return TypeSafeClassifier(api_key=settings.typesafe_api_key)
    return TypeSafeClassifier()


def _build_questions() -> dict[str, Noul | Choice | Score]:
    """Declare all three questions once, for one fan-out request per match.

    Asking independent questions together in a single `.invoke()` call is
    cheaper and faster than three sequential calls, since Jev answers all of
    them in one parallel pass.
    """
    return {
        "overall_fit": Score(
            instructions=(
                "Based on the resume and job description, how strong is "
                "this candidate's overall fit for the role?"
            ),
            criteria=_OVERALL_FIT_CRITERIA,
        ),
        "has_required_skills": Noul(
            instructions=(
                "Does the resume mention the core technical skills listed "
                "in the job description?"
            ),
        ),
        "seniority_match": Choice(
            instructions=(
                "How does the candidate's seniority compare to what the "
                "job description asks for?"
            ),
            criteria=_SENIORITY_CRITERIA,
        ),
    }


def _to_match_result(response: ClassifierResponse) -> MatchResult:
    """Convert a raw TypeSafeClassifier response into a MatchResult."""
    fit = response.scores["overall_fit"]
    skills = response.nouls["has_required_skills"]
    seniority = response.choices["seniority_match"]

    max_level = len(_OVERALL_FIT_CRITERIA) - 1
    # fit.score is a probability-weighted mean level index in [0, max_level];
    # normalize to [0, 1] for the API response.
    normalized_fit_score = fit.score / max_level
    # fit.legend maps every level index to its criteria text; take the one
    # closest to the (possibly fractional) score as the human-readable summary.
    closest_level = min(max(round(fit.score), 0), max_level)
    fit_legend = str(fit.legend[closest_level])

    return MatchResult(
        fit_score=normalized_fit_score,
        fit_confidence=fit.confidence,
        fit_legend=fit_legend,
        has_required_skills=skills.noul >= 0.5,
        has_required_skills_probability=skills.noul,
        seniority_match=seniority.choice,
        seniority_confidence=seniority.confidence,
        model=response.model,
        request_id=response.request_id,
    )


def match_resume(resume_text: str, job_description_text: str) -> MatchResult:
    """Score a single resume against a single job description."""
    classifier = get_classifier()
    response = classifier.invoke(_build_request(resume_text, job_description_text))
    return _to_match_result(response)


def match_resumes(
    resume_texts: list[str], job_description_text: str
) -> list[MatchResult]:
    """Score many resumes against one job description in a single batch call.

    Uses TypeSafeClassifier.batch(), inherited from the LangChain Runnable
    interface, instead of looping over match_resume() -- this lets the
    underlying client parallelize the requests instead of running them one
    at a time.
    """
    classifier = get_classifier()
    inputs: list[ClassifierRequest] = [
        _build_request(resume_text, job_description_text)
        for resume_text in resume_texts
    ]
    responses = classifier.batch(inputs)
    return [_to_match_result(response) for response in responses]
