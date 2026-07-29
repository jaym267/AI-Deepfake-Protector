/**
 * Application shell and upload→poll→result state machine.
 *
 * One file at a time, one result on screen. No history, no accounts, no gallery
 * of past checks — the privacy commitment (docs/DECISIONS.md D2) is that an
 * upload doesn't outlive its analysis, and a UI that accumulates a person's
 * results would quietly work against that.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, fetchLimits, pollUntilDone, startAnalysis } from "./api/client";
import type { AnalysisResult, AnalysisStatus, Limits, MediaKind } from "./api/types";
import { AnalysisProgress } from "./components/AnalysisProgress";
import { ResultsDashboard } from "./components/ResultsDashboard";
import { UploadPanel } from "./components/UploadPanel";

type Phase =
  | { name: "idle" }
  | { name: "working"; kind: MediaKind; filename: string; status: AnalysisStatus }
  | { name: "done"; result: AnalysisResult; filename: string }
  | { name: "error"; message: string };

export default function App() {
  const [limits, setLimits] = useState<Limits | null>(null);
  const [phase, setPhase] = useState<Phase>({ name: "idle" });
  const abortRef = useRef<AbortController | null>(null);

  // Fetched rather than hardcoded so the caps shown to the user are always the
  // ones the server will actually enforce.
  useEffect(() => {
    const controller = new AbortController();
    fetchLimits(controller.signal)
      .then(setLimits)
      .catch(() => {
        // Non-fatal: uploads still work, the server just gets to reject
        // oversized files instead of the client warning first.
      });
    return () => controller.abort();
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  const onSubmit = useCallback(async (file: File, kind: MediaKind) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase({ name: "working", kind, filename: file.name, status: "pending" });

    try {
      const accepted = await startAnalysis(kind, file, controller.signal);

      const final = await pollUntilDone(
        accepted.analysis_id,
        (tick) =>
          setPhase((current) =>
            current.name === "working" ? { ...current, status: tick.status } : current,
          ),
        controller.signal,
      );

      if (final.status === "failed" || !final.result) {
        setPhase({
          name: "error",
          message:
            final.error ??
            "The analysis didn't complete. Your file has already been deleted; please try again.",
        });
        return;
      }

      setPhase({ name: "done", result: final.result, filename: file.name });
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setPhase({
        name: "error",
        message:
          error instanceof ApiError
            ? error.message
            : "Couldn't reach the analysis server. Check that the backend is running on " +
              "http://localhost:8000 and try again.",
      });
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setPhase({ name: "idle" });
  }, []);

  return (
    <div className="app">
      <header className="masthead">
        <h1>AI Deepfake Protection</h1>
        <p className="masthead__tagline">
          Free, for anyone. Upload an image, a voice recording or a video and get
          a plain-language read on whether it shows signs of AI manipulation —
          with the reasons, and an honest account of how sure it is.
        </p>
      </header>

      <main>
        {phase.name === "idle" && (
          <UploadPanel limits={limits} busy={false} onSubmit={onSubmit} />
        )}

        {phase.name === "working" && (
          <AnalysisProgress
            kind={phase.kind}
            filename={phase.filename}
            status={phase.status}
          />
        )}

        {phase.name === "done" && (
          <ResultsDashboard
            result={phase.result}
            filename={phase.filename}
            onReset={reset}
          />
        )}

        {phase.name === "error" && (
          <section className="panel">
            <h2>That didn't work</h2>
            <p className="notice notice--error" role="alert">
              {phase.message}
            </p>
            <button type="button" className="button button--primary" onClick={reset}>
              Try again
            </button>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>
          This tool gives probabilistic estimates, never proof, and cannot tell
          you who created or altered a file. It is not a forensic examination or
          a legal determination.
        </p>
        <p className="muted">
          Uploads are deleted immediately after analysis and are never used as
          training data.
        </p>
      </footer>
    </div>
  );
}
