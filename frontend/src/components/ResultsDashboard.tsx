/**
 * Results dashboard.
 *
 * Reading order is deliberate: hedged headline, then the caveat, then the
 * evidence, then the disclaimer. The uncertainty is placed *above* the detail
 * rather than in a footnote below it, because a reader who stops after the first
 * line should still have read the qualification.
 *
 * What this component must never render: a numeric score, a percentage, or a
 * threshold. The API doesn't return them (docs/DECISIONS.md D3) and there is
 * nothing to derive one from — that is by design, not an omission to fix.
 */

import { useState } from "react";

import { ApiError, requestReport } from "../api/client";
import { CONFIDENCE_COPY, KIND_LABEL, SIGNAL_COPY, VERDICT_COPY } from "../copy";
import type { AnalysisResult } from "../api/types";
import { EvidenceList } from "./EvidenceList";

interface Props {
  result: AnalysisResult;
  filename: string | null;
  onReset: () => void;
}

export function ResultsDashboard({ result, filename, onReset }: Props) {
  const verdict = VERDICT_COPY[result.verdict];
  const [reportNote, setReportNote] = useState<string | null>(null);
  const [reportPending, setReportPending] = useState(false);

  const onRequestReport = async () => {
    setReportPending(true);
    setReportNote(null);
    try {
      const response = await requestReport(result.analysis_id);
      if (response.status === "ready" && response.download_url) {
        window.location.assign(response.download_url);
      } else {
        setReportNote(response.detail ?? "The report isn't available yet.");
      }
    } catch (error) {
      // 501 is the expected answer until build step 7 — the backend's own detail
      // string explains that, so show it rather than inventing wording here.
      setReportNote(
        error instanceof ApiError
          ? error.message
          : "Couldn't reach the server to build a report.",
      );
    } finally {
      setReportPending(false);
    }
  };

  return (
    <section className="panel" aria-labelledby="result-heading">
      {result.is_mock && (
        <p className="notice notice--mock" role="status">
          <strong>Placeholder result — not a real analysis.</strong> The
          detection models aren't built yet. This page is wired to stand-in
          scores so the interface can be developed. Nothing here says anything
          about the file you uploaded.
        </p>
      )}

      <div className={`verdict verdict--${verdict.tone}`}>
        <p className="verdict__kind">
          {KIND_LABEL[result.media_kind]}
          {filename ? ` · ${filename}` : ""}
        </p>
        <h2 id="result-heading" className="verdict__headline">
          {verdict.headline}
        </h2>
        <p className="verdict__confidence">{CONFIDENCE_COPY[result.confidence]}</p>
        <p className="verdict__detail">{verdict.detail}</p>
      </div>

      <h3>What was found</h3>
      <EvidenceList evidence={result.evidence} />

      {result.signals_used.length > 0 && (
        <>
          <h3>Checks that ran</h3>
          <ul className="signals">
            {result.signals_used.map((signal) => {
              const copy = SIGNAL_COPY[signal];
              return (
                <li key={signal} className="signals__item">
                  <span className="signals__label">{copy?.label ?? signal}</span>
                  {copy && <span className="signals__detail">{copy.detail}</span>}
                </li>
              );
            })}
          </ul>
        </>
      )}

      <div className="result-meta">
        {result.media_deleted && (
          <p className="result-meta__deleted">
            Your file has been deleted from the server.
          </p>
        )}
        <p className="muted">
          Checked {new Date(result.analysed_at).toLocaleString()} · Reference{" "}
          <code>{result.analysis_id}</code> · Results are kept for 24 hours.
        </p>
      </div>

      {/* Served by the backend so one approved wording reaches the dashboard and
          the PDF report together. Never replace this with local copy. */}
      <p className="disclaimer">{result.disclaimer}</p>

      <div className="actions">
        <button type="button" className="button button--primary" onClick={onReset}>
          Check another file
        </button>
        <button
          type="button"
          className="button"
          onClick={onRequestReport}
          disabled={reportPending}
        >
          {reportPending ? "Preparing…" : "Download evidence report (PDF)"}
        </button>
      </div>
      {reportNote && (
        <p className="notice notice--info" role="status">
          {reportNote}
        </p>
      )}
    </section>
  );
}
