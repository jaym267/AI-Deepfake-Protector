/**
 * Client-side file classification and pre-flight validation.
 *
 * The accepted MIME types below mirror `ALLOWED_CONTENT_TYPES` in
 * backend/app/uploads.py, and the byte caps come from GET /limits at runtime
 * rather than being duplicated as constants here.
 *
 * This is a courtesy check, not a security control. It exists so someone does
 * not push 25MB uphill only to be rejected — the server re-validates everything
 * and is the only side that actually enforces anything.
 */

import type { Limits, MediaKind } from "./types";

export const ACCEPTED: Record<MediaKind, string[]> = {
  image: ["image/jpeg", "image/png", "image/webp"],
  audio: [
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/mp4",
    "audio/webm",
    "audio/ogg",
  ],
  video: ["video/mp4", "video/quicktime", "video/webm"],
};

/** The `accept` attribute for the file input: everything the backend takes. */
export const ACCEPT_ATTRIBUTE = Object.values(ACCEPTED).flat().join(",");

export function classify(file: File): MediaKind | null {
  const type = file.type.split(";")[0]?.trim().toLowerCase() ?? "";
  for (const kind of ["image", "audio", "video"] as const) {
    if (ACCEPTED[kind].includes(type)) return kind;
  }
  return null;
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${Math.round(bytes / (1024 * 1024))}MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)}KB`;
  return `${bytes} bytes`;
}

/** Duration caps are server-enforced; this only advertises them up front so
 *  someone doesn't push 25MB uphill to be told the clip is too long. */
export function formatDuration(seconds: number | undefined): string {
  if (seconds === undefined) return "";
  if (seconds < 90) return `${Math.round(seconds)} seconds`;
  const minutes = seconds / 60;
  return Number.isInteger(minutes) ? `${minutes} minutes` : `${minutes.toFixed(1)} minutes`;
}

export interface ValidationFailure {
  message: string;
}

export function validate(
  file: File,
  limits: Limits | null,
): { kind: MediaKind } | ValidationFailure {
  if (file.size === 0) {
    return { message: "That file is empty." };
  }

  const kind = classify(file);
  if (!kind) {
    return {
      message:
        `"${file.name}" isn't a file type this tool can read. ` +
        "Accepted: JPEG, PNG or WebP images; MP3, WAV, M4A, OGG or WebM audio; " +
        "MP4, MOV or WebM video.",
    };
  }

  // Limits unavailable (backend down, /limits failed) — let the server decide
  // rather than inventing a cap the frontend would then have to keep in sync.
  if (!limits) return { kind };

  const cap = limits[kind].max_bytes;
  if (file.size > cap) {
    return {
      message:
        `That ${kind} is ${formatBytes(file.size)}, over the ` +
        `${formatBytes(cap)} limit for ${kind} uploads. Limits are strict while ` +
        "this is in development.",
    };
  }

  return { kind };
}

export function isValid(
  outcome: { kind: MediaKind } | ValidationFailure,
): outcome is { kind: MediaKind } {
  return "kind" in outcome;
}
