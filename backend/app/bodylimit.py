"""Request body ceiling, enforced before anything parses the body.

Why this exists
---------------
``uploads.py`` streams an upload to a temp file under a hard size cap and has
always claimed that cap was applied *while streaming*. It was not. With
``file: UploadFile = File(...)``, Starlette's multipart parser consumes the
entire request body before the endpoint function is ever called, spooling it to
a ``SpooledTemporaryFile`` on disk once it passes ``spool_max_size``. By the
time ``staged_upload`` runs, every byte is already written. The loop there was
copying from a temp file, not from the network, so the cap only ever decided the
status code — never how much disk got used.

Measured before this middleware existed: a 40MB body to ``/analyze/image``
(cap 5MB) returned a correct 413 with all 41,943,048 bytes already spooled to
disk. On an unauthenticated endpoint that is a disk-exhaustion primitive, and
the byte cap that was supposed to prevent it was downstream of the damage.

Two paths, because there are two kinds of client
------------------------------------------------
1. **``Content-Length`` present** — the overwhelming majority, including every
   browser and curl. Rejected outright before the request reaches the app, so
   nothing is read and nothing is buffered.

2. **No ``Content-Length``** (chunked transfer-encoding) — the size is not
   knowable up front, so bytes are counted as they arrive and the body is cut
   off at the ceiling. Truncation alone would be dangerous: a truncated file
   could still parse and produce a *verdict* on partial media, so the request is
   also flagged in ``scope["state"]`` and ``uploads.staged_upload`` refuses it.
   Truncating without that flag would trade a disk-exhaustion bug for a
   wrong-answer bug, which is worse in a tool people act on.

Why ASGI rather than ``BaseHTTPMiddleware``
-------------------------------------------
``BaseHTTPMiddleware`` cannot intercept ``receive``; it only sees a ``Request``
whose body is read downstream. Wrapping ``receive`` directly is the only place
the count can be enforced before the parser sees the bytes.

The limit is per-route because the per-kind caps differ by 5x, and applying the
video cap everywhere would leave ``/analyze/image`` five times more expensive to
attack than its own configured limit allows.
"""

from __future__ import annotations

import logging

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import settings
from .schemas import MediaKind

logger = logging.getLogger(__name__)

#: Multipart framing overhead: boundary lines, per-part headers, trailing CRLFs.
#: The body is always larger than the file inside it, so the ceiling has to be
#: the file cap plus slack or a file at exactly the cap would be rejected.
MULTIPART_OVERHEAD = 64 * 1024

#: Everything that is not a media upload. ``/report`` takes a small JSON body
#: and nothing else accepts one, so a request larger than this to any other
#: route is either broken or hostile. Generous by two orders of magnitude
#: against the ~70-byte body the report endpoint actually receives.
DEFAULT_MAX_BODY = 64 * 1024

#: Set on ``scope["state"]`` when a chunked body was cut off at the ceiling.
#: ``uploads.staged_upload`` reads it via ``request.state``.
TRUNCATED_FLAG = "body_limit_exceeded"

_UPLOAD_PATHS: dict[str, MediaKind] = {
    "/analyze/image": MediaKind.IMAGE,
    "/analyze/audio": MediaKind.AUDIO,
    "/analyze/video": MediaKind.VIDEO,
}


def max_body_for(path: str) -> int:
    """The byte ceiling for a request to ``path``, including framing overhead."""
    kind = _UPLOAD_PATHS.get(path.rstrip("/"))
    if kind is None:
        return DEFAULT_MAX_BODY

    caps = {
        MediaKind.IMAGE: settings.max_image_bytes,
        MediaKind.AUDIO: settings.max_audio_bytes,
        MediaKind.VIDEO: settings.max_video_bytes,
    }
    return caps[kind] + MULTIPART_OVERHEAD


async def _send_413(send: Send, limit: int) -> None:
    # Written by hand rather than via JSONResponse to keep this module free of
    # the routing layer. CORS is registered outside this middleware, so it still
    # attaches its headers to this response — same reasoning as the 429 in
    # ratelimit.py, and asserted by a test for the same reason.
    megabytes = limit // (1024 * 1024)
    body = (
        b'{"detail":"This upload is larger than this service accepts '
        b'(about ' + str(max(megabytes, 1)).encode() + b'MB for this file type). '
        b'Limits are strict while this is in development."}'
    )
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class BodySizeLimitMiddleware:
    """Reject or truncate request bodies over the per-route ceiling."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = max_body_for(scope.get("path", ""))
        headers = Headers(scope=scope)
        declared = headers.get("content-length")

        # Path 1: the client told us up front. Refuse without reading a byte.
        if declared is not None:
            try:
                if int(declared) > limit:
                    logger.info(
                        "Rejected %s: declared body %s bytes over the %d ceiling",
                        scope.get("path"),
                        declared,
                        limit,
                    )
                    await _send_413(send, limit)
                    return
            except ValueError:
                # A malformed Content-Length is the client's problem, but it is
                # not this middleware's job to decide the status code for it.
                # Fall through to counting, which bounds the damage either way.
                pass

        # Path 2: chunked, or a Content-Length that lied. Count as it arrives.
        state = scope.setdefault("state", {})
        state[TRUNCATED_FLAG] = False
        received = 0

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] != "http.request":
                return message

            received += len(message.get("body", b""))
            if received > limit:
                logger.info(
                    "Truncated %s: body exceeded the %d byte ceiling",
                    scope.get("path"),
                    limit,
                )
                state[TRUNCATED_FLAG] = True
                # End the body here. staged_upload refuses the request on the
                # flag, so the truncated bytes are never analysed.
                return {"type": "http.request", "body": b"", "more_body": False}
            return message

        await self.app(scope, counting_receive, send)
