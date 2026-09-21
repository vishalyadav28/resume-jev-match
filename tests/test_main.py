"""API-level tests for app.main, with the classifier layer mocked out.

Settings are injected via FastAPI's dependency_overrides rather than
monkeypatching the environment: Settings reads a local .env file directly
(independent of os.environ), so an env-var-based approach would behave
differently depending on whether a developer's machine happens to have one.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.classifier import MatchResult
from app.config import Settings, get_settings
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


def _use_settings(**overrides: object) -> None:
    def _override() -> Settings:
        return Settings(**overrides)  # type: ignore[arg-type]

    app.dependency_overrides[get_settings] = _override


def teardown_function() -> None:
    app.dependency_overrides.clear()


def test_health() -> None:
    _use_settings(typesafe_api_key="test-key")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "typesafe_configured": True}


def test_match_missing_api_key_returns_503() -> None:
    _use_settings(typesafe_api_key=None)
    response = client.post(
        "/match", json={"resume": "text", "job_description": "text"}
    )
    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == "missing_api_key"


@patch("app.main.match_resume")
def test_match_returns_mapped_response(mock_match_resume) -> None:
    _use_settings(typesafe_api_key="test-key")
    mock_match_resume.return_value = _fake_result()

    response = client.post(
        "/match", json={"resume": "resume text", "job_description": "jd text"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fit_score"] == 0.8
    assert body["needs_human_review"] is False
    assert body["model"] == "jev-2026-09-15"


@patch("app.main.match_resumes")
def test_batch_match_sorted_by_fit_score(mock_match_resumes) -> None:
    _use_settings(typesafe_api_key="test-key")
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


@patch("app.main.match_resume")
def test_low_confidence_triggers_human_review(mock_match_resume) -> None:
    _use_settings(typesafe_api_key="test-key")
    mock_match_resume.return_value = _fake_result(fit_confidence=0.2)

    response = client.post(
        "/match", json={"resume": "resume text", "job_description": "jd text"}
    )

    assert response.json()["needs_human_review"] is True
