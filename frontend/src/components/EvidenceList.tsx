/**
 * The evidence breakdown — the part of the report that does the actual work.
 *
 * A verdict band on its own is a black box asking to be trusted. The point of
 * this product is that a person can read *why*, decide for themselves, and show
 * it to someone else. So each finding renders as a sentence, with where in the
 * file it was found, and no numbers attached.
 */

import { SEVERITY_LABEL } from "../copy";
import type { EvidenceItem } from "../api/types";

function timespan(item: EvidenceItem): string | null {
  const { start_seconds: start, end_seconds: end } = item;
  if (start == null) return null;

  const clock = (s: number) => {
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return end != null && end > start
    ? `${clock(start)}–${clock(end)}`
    : `at ${clock(start)}`;
}

export function EvidenceList({ evidence }: { evidence: EvidenceItem[] }) {
  if (evidence.length === 0) {
    return (
      <p className="muted">No specific findings were recorded for this file.</p>
    );
  }

  return (
    <ul className="evidence">
      {evidence.map((item) => {
        const span = timespan(item);
        return (
          <li key={item.code} className={`evidence__item evidence__item--${item.severity}`}>
            <div className="evidence__head">
              <span className={`chip chip--${item.severity}`}>
                {SEVERITY_LABEL[item.severity]}
              </span>
              {(span || item.region) && (
                <span className="evidence__where">
                  {[span, item.region].filter(Boolean).join(" · ")}
                </span>
              )}
            </div>
            <p className="evidence__summary">{item.summary}</p>
          </li>
        );
      })}
    </ul>
  );
}
