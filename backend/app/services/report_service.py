"""Evidence report generation — STUB (real implementation is build step 7).

The endpoint and its contract exist now so the frontend can wire up the
"Download evidence report" action against the real shape during step 2. It
returns 501 with a clear reason until the real generator lands.

Planned implementation (step 7), in order:
  1. Pull the stored analysis record by id — never trust a client-supplied
     result body (see schemas.ReportRequest for why).
  2. Pass the *internal* technical outputs to an LLM to draft plain-language
     explanations. This is the LLM's only role in the system: translating
     findings the models produced into prose. It never decides the verdict.
  3. Render to PDF with the verdict band, the evidence list with timestamps, the
     analysis id and timestamp, and REPORT_DISCLAIMER on every page.

The report must not embed per-model numeric scores: it is a document explicitly
intended to be forwarded to platforms and employers, which makes it the least
controllable surface in the product.
"""

from __future__ import annotations

from ..schemas import AnalysisJob, ReportResponse


def generate_report(job: AnalysisJob) -> ReportResponse:
    return ReportResponse(
        analysis_id=job.analysis_id,
        format="pdf",
        status="not_implemented",
        download_url=None,
        detail=(
            "PDF evidence reports are not available yet. They are build step 7, "
            "scheduled after the real detection models are in place."
        ),
    )
