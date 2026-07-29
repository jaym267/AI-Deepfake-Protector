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

    # --- Rate limiting (D5 item 3) ---
    # /analyze accepts a 25MB upload and runs inference on it, so writes are
    # capped hard and reads are not: the frontend polls ~1/s while a job runs
    # (D1), and a limit tuned for uploads would break normal use.
    #
    # Two windows because one is always wrong — a burst limit alone misses a
    # slow drip, a sustained limit alone lets through a damaging spike.
    rate_limit_enabled: bool = True
    analyze_burst_limit: int = 5
    analyze_burst_window: int = 60
    analyze_sustained_limit: int = 30
    analyze_sustained_window: int = 3600
    read_limit: int = 300
    read_window: int = 60

    # Off by default: any client can send X-Forwarded-For, so honouring it on a
    # directly-exposed server makes the limiter bypassable with one curl flag.
    # Turn on only behind a proxy that overwrites the header.
    trust_forwarded_for: bool = False

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


settings = Settings()
