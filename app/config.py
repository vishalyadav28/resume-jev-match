"""Typed application configuration, loaded once from the environment."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, validated at startup instead of read ad hoc.

    TypeSafeClassifier itself already reads TYPESAFE_API_KEY from the
    environment, so that field exists here mainly so `/health` can report
    whether a key is configured without constructing a classifier.
    `confidence_gate_threshold` is the one true tunable: below it, a match
    response is flagged `needs_human_review`.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    typesafe_api_key: str | None = Field(default=None, alias="TYPESAFE_API_KEY")
    confidence_gate_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
