"""API-level tests for app.main, with the classifier layer mocked out."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.classifier import MatchResult
from app.main import app

client = TestClient(app)


def _fake_result(fit_score: float = 0.8, fit_confidence: float = 0.9) -> MatchResult:
    return MatchResult(
        fit_score=fit_score,
        fit_confidence=fit_confidence,
        fit_legend="Strong match.",
        has_required_skills=True,
        has_required_skills_probability=0.9,
        seniority_match="matched",
        seniority_confidence=0.9,
        model="jev-2026-09-15",
        request_id="req-123",
    )


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_match_missing_api_key_returns_503(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    from app.config import get_settings

    get_settings.cache_clear()
    response = client.post(
        "/match", json={"resume": "text", "job_description": "text"}
    )
    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == "missing_api_key"
    get_settings.cache_clear()


@patch("app.main.match_resume")
def test_match_returns_mapped_response(mock_match_resume, monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    from app.config import get_settings

    get_settings.cache_clear()
    mock_match_resume.return_value = _fake_result()

    response = client.post(
        "/match", json={"resume": "resume text", "job_description": "jd text"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fit_score"] == 0.8
    assert body["needs_human_review"] is False
    assert body["model"] == "jev-2026-09-15"
    get_settings.cache_clear()


@patch("app.main.match_resumes")
def test_batch_match_sorted_by_fit_score(mock_match_resumes, monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    from app.config import get_settings

    get_settings.cache_clear()
    mock_match_resumes.return_value = [
        _fake_result(fit_score=0.3),
        _fake_result(fit_score=0.9),
    ]

    response = client.post(
        "/batch-match",
        json={
            "resumes": [{"id": "low", "resume": "a"}, {"id": "high", "resume": "b"}],
            "job_description": "jd text",
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["resume_id"] for r in results] == ["high", "low"]
    get_settings.cache_clear()


@patch("app.main.match_resume")
def test_low_confidence_triggers_human_review(mock_match_resume, monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    from app.config import get_settings

    get_settings.cache_clear()
    mock_match_resume.return_value = _fake_result(fit_confidence=0.2)

    response = client.post(
        "/match", json={"resume": "resume text", "job_description": "jd text"}
    )

    assert response.json()["needs_human_review"] is True
    get_settings.cache_clear()
