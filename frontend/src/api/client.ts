/**
 * Backend client.
 *
 * The upload flow is async by contract (docs/DECISIONS.md D1): POST returns 202
 * with a poll_url, and the client polls until the status is terminal. The stubs
 * finish inline today, so the first poll almost always returns `complete` — the
 * loop exists anyway so that nothing here changes when step 3 moves inference
 * onto a worker queue and a video starts taking 30 seconds.
 */

import type {
  AnalysisAccepted,
  AnalysisStatusResponse,
  Limits,
  MediaKind,
  ReportResponse,
} from "./types";

export const API_BASE =
  import.meta.env.VITE_API_BASE?.replace(/\/$/, "") ?? "http://localhost:8000";

/** An error carrying the backend's own `detail` string, which is written to be
 *  shown to a user (size caps, unsupported types, expired ids). */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseError(response: Response): Promise<never> {
  let detail: string | undefined;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // Non-JSON error body (proxy timeout, HTML error page). Fall through.
  }
  throw new ApiError(
    response.status,
    detail ?? `Request failed (${response.status} ${response.statusText}).`,
  );
}

export async function fetchLimits(signal?: AbortSignal): Promise<Limits> {
  const response = await fetch(`${API_BASE}/limits`, { signal });
  if (!response.ok) return parseError(response);
  return (await response.json()) as Limits;
}

export async function startAnalysis(
  kind: MediaKind,
  file: File,
  signal?: AbortSignal,
): Promise<AnalysisAccepted> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE}/analyze/${kind}`, {
    method: "POST",
    body,
    signal,
  });
  if (!response.ok) return parseError(response);
  return (await response.json()) as AnalysisAccepted;
}

export async function fetchAnalysis(
  analysisId: string,
  signal?: AbortSignal,
): Promise<AnalysisStatusResponse> {
  const response = await fetch(`${API_BASE}/analyze/${analysisId}`, { signal });
  if (!response.ok) return parseError(response);
  return (await response.json()) as AnalysisStatusResponse;
}

export async function requestReport(
  analysisId: string,
  signal?: AbortSignal,
): Promise<ReportResponse> {
  const response = await fetch(`${API_BASE}/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ analysis_id: analysisId, format: "pdf" }),
    signal,
  });
  // 501 until step 7 lands. Surfaced as an ApiError and handled by the caller.
  if (!response.ok) return parseError(response);
  return (await response.json()) as ReportResponse;
}

const POLL_INTERVAL_MS = 1_000;
const POLL_TIMEOUT_MS = 120_000;

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });

/**
 * Poll until the analysis reaches a terminal state.
 *
 * The timeout is a client-side guard only — it stops the spinner, it does not
 * cancel server work. Real cancellation needs a DELETE endpoint, which is worth
 * adding when inference actually costs GPU time (step 3).
 */
export async function pollUntilDone(
  analysisId: string,
  onTick?: (status: AnalysisStatusResponse) => void,
  signal?: AbortSignal,
): Promise<AnalysisStatusResponse> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;

  for (;;) {
    const status = await fetchAnalysis(analysisId, signal);
    onTick?.(status);

    if (status.status === "complete" || status.status === "failed") {
      return status;
    }
    if (Date.now() > deadline) {
      throw new ApiError(
        504,
        "This analysis is taking longer than expected. Your file has already been " +
          "deleted from the server; please try again.",
      );
    }
    await sleep(POLL_INTERVAL_MS, signal);
  }
}
