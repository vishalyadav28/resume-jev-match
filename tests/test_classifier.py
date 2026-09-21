"""Unit tests for app.classifier, with TypeSafeClassifier mocked out.

These tests never hit the network: they assert that classifier.py builds the
right request shape and correctly maps Jev's nested response fields onto
MatchResult, independent of whether Jev itself is reachable.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.classifier import match_resume, match_resumes


def _fake_response(
    score: float = 1.5,
    fit_confidence: float = 0.8,
    skill_probability: float = 0.9,
    choice: str = "matched",
    choice_confidence: float = 0.7,
) -> SimpleNamespace:
    return SimpleNamespace(
        scores={
            "overall_fit": SimpleNamespace(
                score=score, confidence=fit_confidence, legend="Strong match."
            )
        },
        nouls={"has_required_skills": SimpleNamespace(noul=skill_probability)},
        choices={
            "seniority_match": SimpleNamespace(
                choice=choice, confidence=choice_confidence
            )
        },
        model="jev-2026-09-15",
        request_id="req-123",
    )


@patch("app.classifier.get_classifier")
def test_match_resume_maps_nested_fields(mock_get_classifier: MagicMock) -> None:
    mock_classifier = MagicMock()
    mock_classifier.invoke.return_value = _fake_response()
    mock_get_classifier.return_value = mock_classifier

    result = match_resume("resume text", "job description text")

    assert result.fit_score == 0.75  # score=1.5 normalized over 2 steps
    assert result.fit_confidence == 0.8
    assert result.has_required_skills is True
    assert result.has_required_skills_probability == 0.9
    assert result.seniority_match == "matched"
    assert result.model == "jev-2026-09-15"
    assert result.request_id == "req-123"


@patch("app.classifier.get_classifier")
def test_match_resume_skill_below_threshold_is_false(
    mock_get_classifier: MagicMock,
) -> None:
    mock_classifier = MagicMock()
    mock_classifier.invoke.return_value = _fake_response(skill_probability=0.2)
    mock_get_classifier.return_value = mock_classifier

    result = match_resume("resume text", "job description text")

    assert result.has_required_skills is False


@patch("app.classifier.get_classifier")
def test_match_resume_sends_one_fan_out_request(
    mock_get_classifier: MagicMock,
) -> None:
    mock_classifier = MagicMock()
    mock_classifier.invoke.return_value = _fake_response()
    mock_get_classifier.return_value = mock_classifier

    match_resume("resume text", "job description text")

    assert mock_classifier.invoke.call_count == 1
    call_kwargs = mock_classifier.invoke.call_args[0][0]
    assert set(call_kwargs["questions"]) == {
        "overall_fit",
        "has_required_skills",
        "seniority_match",
    }


@patch("app.classifier.get_classifier")
def test_match_resumes_uses_batch(mock_get_classifier: MagicMock) -> None:
    mock_classifier = MagicMock()
    mock_classifier.batch.return_value = [_fake_response(), _fake_response()]
    mock_get_classifier.return_value = mock_classifier

    results = match_resumes(["resume one", "resume two"], "job description text")

    assert len(results) == 2
    assert mock_classifier.batch.call_count == 1
    assert mock_classifier.invoke.call_count == 0
