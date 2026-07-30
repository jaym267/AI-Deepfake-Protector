"""Domain exceptions that carry a user-safe message.

The point of these is that the router can map them to a status code without
needing a broad ``except``. That matters more than it looks: the previous
handler caught bare ``ValueError`` and reflected ``str(exc)`` into a 400 body,
so any ValueError raised anywhere inside PIL, numpy or torch would have had its
text — potentially including a filesystem path — returned to the caller. An
internal invariant failing is a 500, and it should not be able to dress itself
up as a validation error.

Every message on these is written to be read by the person who uploaded the
file, not by whoever is reading the logs.
"""

from __future__ import annotations


class UndecodableUpload(ValueError):
    """The bytes are not the media they claimed to be, or cannot be decoded.

    Subclasses ValueError only so existing call sites that catch ValueError keep
    working; the router matches on this type specifically.
    """


class UploadTooLarge(ValueError):
    """Decoded dimensions or duration exceed what this service will process.

    Distinct from ``UndecodableUpload`` because the file is perfectly valid — it
    is just larger than the service is willing to spend memory on, which is a
    413 rather than a 400.
    """
