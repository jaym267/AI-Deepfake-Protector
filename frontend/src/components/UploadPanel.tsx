/**
 * Upload surface: drag-and-drop or file picker, one file at a time.
 *
 * A single input for all three media types rather than three separate uploaders.
 * The person using this has "a file someone sent me" — asking them to first
 * classify it as image/audio/video is a step they shouldn't have to take, and
 * the MIME type already answers it.
 */

import { useCallback, useRef, useState } from "react";
import type { DragEvent } from "react";

import {
  ACCEPT_ATTRIBUTE,
  formatBytes,
  formatDuration,
  isValid,
  validate,
} from "../api/media";
import type { Limits, MediaKind } from "../api/types";

interface Props {
  limits: Limits | null;
  busy: boolean;
  onSubmit: (file: File, kind: MediaKind) => void;
}

export function UploadPanel({ limits, busy, onSubmit }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const accept = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      const outcome = validate(file, limits);
      if (!isValid(outcome)) {
        setError(outcome.message);
        return;
      }
      setError(null);
      onSubmit(file, outcome.kind);
    },
    [limits, onSubmit],
  );

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    if (busy) return;
    accept(event.dataTransfer.files[0]);
  };

  return (
    <section className="panel" aria-labelledby="upload-heading">
      <h2 id="upload-heading">Check a file</h2>

      <div
        className={`dropzone${dragging ? " dropzone--active" : ""}${busy ? " dropzone--busy" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!busy) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <p className="dropzone__lead">Drop a file here</p>
        <p className="dropzone__or">or</p>
        <button
          type="button"
          className="button button--primary"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
        >
          Choose a file
        </button>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTRIBUTE}
          className="visually-hidden"
          disabled={busy}
          onChange={(e) => {
            accept(e.target.files?.[0]);
            // Reset so re-picking the same file fires onChange again.
            e.target.value = "";
          }}
        />

        <p className="dropzone__formats">
          Images (JPEG, PNG, WebP) · Audio (MP3, WAV, M4A, OGG, WebM) · Video
          (MP4, MOV, WebM)
        </p>
        {limits && (
          <p className="dropzone__limits">
            Up to {formatBytes(limits.image.max_bytes)} for images;{" "}
            {formatBytes(limits.audio.max_bytes)} and{" "}
            {formatDuration(limits.audio.max_seconds)} for audio;{" "}
            {formatBytes(limits.video.max_bytes)} and{" "}
            {formatDuration(limits.video.max_seconds)} for video.
          </p>
        )}
      </div>

      {error && (
        <p className="notice notice--error" role="alert">
          {error}
        </p>
      )}

      <p className="privacy-note">
        <strong>Your file is deleted as soon as the check finishes.</strong> It
        is never stored, never used to train anything, and never shared. Only the
        result stays, for 24 hours, so you can come back to it.
      </p>
    </section>
  );
}
