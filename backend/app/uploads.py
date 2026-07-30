"""Upload intake: validation, bounded spooling to disk, guaranteed cleanup.

Rules this module enforces:

1. The size cap is re-checked here while copying, and the temp file never grows
   past it.

   This module used to claim the cap was applied "while streaming, not after",
   and that reading the whole body first "would let a single request exhaust
   memory or disk regardless of the configured limit". The second half was
   correct and described what this code was actually doing. With
   ``UploadFile = File(...)``, Starlette parses the entire multipart body before
   the endpoint runs, so the loop below reads from a temp file Starlette already
   wrote — it was never reading from the network. The cap decided the status code
   and nothing else.

   The ceiling that actually bounds what reaches disk now lives in
   ``bodylimit.py``, which runs as middleware before any parsing. The check here
   is kept as a second line of defence: it is the only one that knows the size of
   the *file* rather than the size of the request that carried it.

2. The temp file is always removed, on every exit path, including errors. The
   privacy commitment is that an upload does not outlive its analysis.

3. The container is probed exactly once. The probe result travels with the
   staged upload rather than being recomputed downstream, so the pipeline cannot
   disagree with the validator about what is in the file — and so attacker
   bytes are handed to ffmpeg's demuxers once per upload instead of twice.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile, status

from .bodylimit import TRUNCATED_FLAG
from .config import settings
from .schemas import MediaKind

CHUNK = 1024 * 1024

# Deliberately conservative allowlists. Anything not listed is rejected rather
# than best-effort decoded.
ALLOWED_CONTENT_TYPES: dict[MediaKind, set[str]] = {
    MediaKind.IMAGE: {"image/jpeg", "image/png", "image/webp"},
    MediaKind.AUDIO: {"audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/webm", "audio/ogg"},
    MediaKind.VIDEO: {"video/mp4", "video/quicktime", "video/webm"},
}

MAX_BYTES: dict[MediaKind, int] = {
    MediaKind.IMAGE: settings.max_image_bytes,
    MediaKind.AUDIO: settings.max_audio_bytes,
    MediaKind.VIDEO: settings.max_video_bytes,
}


@dataclass
class StagedUpload:
    path: Path
    size_bytes: int
    filename: str | None
    content_type: str | None

    #: Whether the container carries an audio stream, from the single probe in
    #: ``_enforce_duration``. ``None`` for images, which are never probed.
    #: Consumed by the pipeline so a silent video drops the audio term from
    #: fusion instead of folding in a score for a track that isn't there.
    has_audio: bool | None = None


def _human_mb(n: int) -> str:
    return f"{n / (1024 * 1024):.0f}MB"


def validate_content_type(kind: MediaKind, content_type: str | None) -> None:
    allowed = ALLOWED_CONTENT_TYPES[kind]
    # Strip any parameters, e.g. "audio/webm; codecs=opus".
    base = (content_type or "").split(";")[0].strip().lower()
    if base not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported {kind.value} type '{base or 'unknown'}'. "
                f"Accepted: {', '.join(sorted(allowed))}."
            ),
        )


@contextlib.contextmanager
def staged_upload(
    kind: MediaKind, upload: UploadFile, request: Request | None = None
) -> Iterator[StagedUpload]:
    """Copy an upload to a temp file under a hard size cap, then always delete it."""
    validate_content_type(kind, upload.content_type)
    _reject_if_truncated(request)
    limit = MAX_BYTES[kind]

    fd, tmp_name = tempfile.mkstemp(prefix="adp_", suffix=_suffix_for(upload.filename))
    tmp_path = Path(tmp_name)
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = upload.file.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"{kind.value.capitalize()} exceeds the "
                            f"{_human_mb(limit)} limit for this MVP."
                        ),
                    )
                out.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        # Duration cap. Deliberately after the file is fully staged: the
        # container metadata that carries duration can sit at either end of the
        # file depending on how it was written, so there is nothing reliable to
        # check mid-stream. The middleware ceiling is what bounds the damage
        # until this point.
        probe = _enforce_duration(kind, tmp_path)

        yield StagedUpload(
            path=tmp_path,
            size_bytes=size,
            filename=upload.filename,
            content_type=upload.content_type,
            has_audio=None if kind is MediaKind.IMAGE else probe.has_audio,
        )
    finally:
        # Runs on success, on validation failure, and on unexpected errors.
        tmp_path.unlink(missing_ok=True)


MAX_SECONDS: dict[MediaKind, float | None] = {
    MediaKind.IMAGE: None,  # no duration
    MediaKind.AUDIO: settings.max_audio_seconds,
    MediaKind.VIDEO: settings.max_video_seconds,
}


def _reject_if_truncated(request: Request | None) -> None:
    """Refuse a body the middleware had to cut off at the ceiling.

    Only reachable for a chunked request with no ``Content-Length``, which
    ``bodylimit`` cannot reject up front. Truncated bytes must never be analysed:
    a short file can still decode, and a verdict on partial media is a wrong
    answer rather than an error, which is the worse failure for this product.
    """
    if request is None:
        return
    if getattr(request.state, TRUNCATED_FLAG, False):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "This upload is larger than this service accepts. "
                "Limits are strict while this is in development."
            ),
        )


def _enforce_duration(kind: MediaKind, path: Path):
    """Reject over-long or unparseable media, as an HTTPException.

    Returns the probe result so the caller can pass it downstream instead of
    opening the file with ffmpeg a second time.

    Translated here rather than in media_probe so that module stays free of web
    framework types and remains usable from the training code.
    """
    from .media_probe import DurationExceeded, UnreadableMedia, enforce_limits

    try:
        return enforce_limits(kind, path, MAX_SECONDS[kind])
    except DurationExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from None
    except UnreadableMedia as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None


def _suffix_for(filename: str | None) -> str:
    if not filename:
        return ""
    suffix = Path(filename).suffix
    # Never reflect an arbitrary client-supplied extension onto the filesystem.
    return suffix if len(suffix) <= 8 and suffix.isascii() else ""
