"""Upload intake: validation, bounded spooling to disk, guaranteed cleanup.

Two rules this module enforces:

1. The size cap is applied *while streaming*, not after. Trusting
   ``Content-Length`` or reading the whole body first would let a single request
   exhaust memory or disk regardless of the configured limit.
2. The temp file is always removed, on every exit path, including errors. The
   privacy commitment is that an upload does not outlive its analysis.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

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
def staged_upload(kind: MediaKind, upload: UploadFile) -> Iterator[StagedUpload]:
    """Stream an upload to a temp file under a hard size cap, then always delete it."""
    validate_content_type(kind, upload.content_type)
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

        # TODO(step 3-5): duration limits (audio 2min / video 60s) need ffprobe or
        # librosa to read container metadata. Deferred until the real decoding
        # pipeline exists so the skeleton has no binary dependency. The byte caps
        # above are the effective limit for now.
        yield StagedUpload(
            path=tmp_path,
            size_bytes=size,
            filename=upload.filename,
            content_type=upload.content_type,
        )
    finally:
        # Runs on success, on validation failure, and on unexpected errors.
        tmp_path.unlink(missing_ok=True)


def _suffix_for(filename: str | None) -> str:
    if not filename:
        return ""
    suffix = Path(filename).suffix
    # Never reflect an arbitrary client-supplied extension onto the filesystem.
    return suffix if len(suffix) <= 8 and suffix.isascii() else ""
