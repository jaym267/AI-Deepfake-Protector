/**
 * Mirrors backend/app/schemas.py.
 *
 * Only the *public* half is modelled here, deliberately. `InternalScores`
 * (per-model numeric scores, the fused score, thresholds) has no counterpart in
 * this file and must not be given one: the two-tier disclosure decision
 * (docs/DECISIONS.md D3) says those never cross the network boundary. If a
 * future response ever carries them, the fix is on the server, not a type here.
 *
 * When schemas.py changes, this file changes with it. There is no codegen step
 * yet; if the contract starts drifting, generate these from the OpenAPI schema
 * at http://localhost:8000/openapi.json rather than hand-patching.
 */

export type MediaKind = "image" | "audio" | "video";

export type AnalysisStatus = "pending" | "running" | "complete" | "failed";

/** Four bands, never a binary real/fake. See docs/DECISIONS.md D4. */
export type Verdict =
  | "likely_authentic"
  | "uncertain"
  | "possibly_manipulated"
  | "likely_manipulated";

/** Banded, not a percentage — a precise number reads as more authoritative
 *  than the models warrant, and is a sharper gradient to tune a fake against. */
export type ConfidenceBand = "low" | "moderate" | "high";

export type Severity = "info" | "notable" | "strong";

export interface EvidenceItem {
  code: string;
  summary: string;
  severity: Severity;
  start_seconds: number | null;
  end_seconds: number | null;
  region: string | null;
}

export interface AnalysisResult {
  analysis_id: string;
  media_kind: MediaKind;
  verdict: Verdict;
  confidence: ConfidenceBand;
  evidence: EvidenceItem[];
  signals_used: string[];
  analysed_at: string;
  media_deleted: boolean;
  /** Authored and approved server-side so one wording ships everywhere.
   *  Always render this value; never hardcode disclaimer copy in the UI. */
  disclaimer: string;
  /** True while the detectors are stubs. Drives the "not a real result" banner. */
  is_mock: boolean;
}

export interface AnalysisAccepted {
  analysis_id: string;
  status: AnalysisStatus;
  poll_url: string;
}

export interface AnalysisStatusResponse {
  analysis_id: string;
  status: AnalysisStatus;
  result: AnalysisResult | null;
  error: string | null;
}

export interface MediaLimits {
  max_bytes: number;
  max_seconds?: number;
}

export interface Limits {
  image: MediaLimits;
  audio: MediaLimits;
  video: MediaLimits;
}

export interface ReportResponse {
  analysis_id: string;
  format: "pdf";
  status: "ready" | "not_implemented";
  download_url: string | null;
  detail: string | null;
}
