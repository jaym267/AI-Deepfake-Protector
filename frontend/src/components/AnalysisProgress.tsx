/**
 * Progress state while an analysis runs.
 *
 * With stub models this is on screen for a fraction of a second. It is built
 * properly anyway: once real inference lands (steps 3-6) a video will sit here
 * for tens of seconds, and this is the moment someone is anxious about a file
 * concerning them — so it names what is happening rather than spinning silently.
 */

import { KIND_LABEL } from "../copy";
import type { AnalysisStatus, MediaKind } from "../api/types";

interface Props {
  kind: MediaKind;
  filename: string | null;
  status: AnalysisStatus;
}

const STAGE_COPY: Record<AnalysisStatus, string> = {
  pending: "Queued…",
  running: "Analysing…",
  complete: "Finishing up…",
  failed: "Something went wrong.",
};

export function AnalysisProgress({ kind, filename, status }: Props) {
  const steps =
    kind === "video"
      ? ["Visual artifacts", "Voice authenticity", "Motion over time", "Combined judgement"]
      : kind === "audio"
        ? ["Voice authenticity"]
        : ["Visual artifacts"];

  return (
    <section className="panel" aria-labelledby="progress-heading">
      <h2 id="progress-heading">{STAGE_COPY[status]}</h2>
      <p className="muted">
        {KIND_LABEL[kind]}
        {filename ? ` · ${filename}` : ""}
      </p>

      <div className="progress-bar" role="progressbar" aria-label="Analysis in progress">
        <div className="progress-bar__fill" />
      </div>

      <ul className="progress-steps">
        {steps.map((step) => (
          <li key={step}>{step}</li>
        ))}
      </ul>

      <p className="muted">
        Your file is deleted as soon as this finishes.
      </p>
    </section>
  );
}
