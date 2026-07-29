/**
 * All user-facing wording for verdicts, confidence and signals, in one place.
 *
 * Centralised because this is the copy that decides whether the product keeps
 * its promise. Two rules it has to hold to:
 *
 *  1. No verdict is stated as fact. Every headline is hedged ("likely",
 *     "signs of") because the underlying result is a probability. A confident
 *     false positive about a real, named person is itself a reputational harm —
 *     which is the exact harm this tool exists to reduce.
 *  2. No numbers. Not scores, not percentages, not thresholds. See
 *     docs/DECISIONS.md D3 — a precise figure is both falsely authoritative and
 *     a useful gradient for anyone tuning a fake to slip past the detector.
 *
 * The disclaimer is deliberately NOT here: it is served by the backend
 * (backend/app/services/disclaimer.py) so a single approved wording ships to the
 * dashboard and the PDF report at once, and cannot drift between them.
 */

import type { ConfidenceBand, MediaKind, Severity, Verdict } from "./api/types";

export interface VerdictCopy {
  headline: string;
  detail: string;
  /** Drives colour/emphasis only. Never rendered as text. */
  tone: "authentic" | "uncertain" | "caution" | "alert";
}

export const VERDICT_COPY: Record<Verdict, VerdictCopy> = {
  likely_authentic: {
    headline: "No clear signs of manipulation",
    detail:
      "The checks that ran didn't find the patterns typically left behind by " +
      "AI generation or editing. That is not the same as proof it's genuine — " +
      "a well-made fake can pass these checks.",
    tone: "authentic",
  },
  uncertain: {
    headline: "Not enough to call either way",
    detail:
      "The signals disagreed, or were too weak to lean on. This is a common " +
      "and legitimate outcome, especially with short, heavily compressed, or " +
      "low-resolution files. It is not a hint in either direction.",
    tone: "uncertain",
  },
  possibly_manipulated: {
    headline: "Some signs of manipulation",
    detail:
      "Patterns consistent with AI generation or editing showed up, but not " +
      "strongly or consistently enough to be confident. Weigh this alongside " +
      "where the file came from and what else you can verify.",
    tone: "caution",
  },
  likely_manipulated: {
    headline: "Strong signs of manipulation",
    detail:
      "Several independent checks found patterns characteristic of AI " +
      "generation or editing. This is still an estimate, not proof, and it " +
      "does not identify who made or altered the file.",
    tone: "alert",
  },
};

export const CONFIDENCE_COPY: Record<ConfidenceBand, string> = {
  low: "Low confidence — treat this as weak information on its own.",
  moderate: "Moderate confidence — worth weighing, but not on its own.",
  high: "High confidence, for an automated check — still not proof.",
};

/** Plain-language names for `signals_used`, so the breakdown is legible to a
 *  non-expert. Which checks ran is safe to disclose; what they scored is not. */
export const SIGNAL_COPY: Record<string, { label: string; detail: string }> = {
  image: {
    label: "Visual artifacts",
    detail:
      "Looks at individual frames for blending seams, warped features and the " +
      "texture patterns AI image generators tend to leave.",
  },
  audio: {
    label: "Voice authenticity",
    detail:
      "Checks the speech for the characteristics of synthetic or cloned voices.",
  },
  raw_frames: {
    label: "Motion over time",
    detail:
      "Compares frames in sequence for flicker, unnatural blinking and " +
      "lighting that shifts in ways real footage doesn't.",
  },
  video_authenticator: {
    label: "Combined judgement",
    detail:
      "Weighs the checks above against each other to reach the overall result, " +
      "rather than trusting any single one.",
  },
};

export const KIND_LABEL: Record<MediaKind, string> = {
  image: "Image",
  audio: "Audio",
  video: "Video",
};

export const SEVERITY_LABEL: Record<Severity, string> = {
  info: "No issue found",
  notable: "Worth noting",
  strong: "Significant",
};
