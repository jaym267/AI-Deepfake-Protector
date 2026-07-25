"""Runtime configuration and upload limits.

Limits are deliberately strict for the MVP to keep compute/storage cheap during
development (see docs/CLAUDE_CODE_BRIEF.md, "Upload limits").
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

MB = 1024 * 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ADP_", env_file=".env", extra="ignore")

    # --- Upload limits (MVP) ---
    max_image_bytes: int = 5 * MB
    max_audio_bytes: int = 10 * MB
    max_video_bytes: int = 25 * MB

    max_audio_seconds: float = 120.0
    max_video_seconds: float = 60.0

    # --- Privacy ---
    # The uploaded file is deleted as soon as scores have been extracted from it.
    # Only the derived JSON result (and, later, evidence thumbnails) is retained.
    # This is a product commitment, not just an implementation detail — see
    # docs/DECISIONS.md.
    delete_upload_after_analysis: bool = True

    # How long a finished analysis result stays queryable before it is dropped
    # from the job store. The uploaded media is already gone by this point.
    result_ttl_seconds: int = 24 * 60 * 60

    # --- Disclosure ---
    # When False (the default, and the only correct setting for the public
    # deployment), per-model numeric scores are never serialised to a client.
    # Exposing them would let someone iterate on a deepfake until it passes.
    expose_internal_scores: bool = False

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


settings = Settings()
