"""Evidence report endpoint. Returns 501 until build step 7."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..schemas import AnalysisStatus, ReportRequest, ReportResponse
from ..services.report_service import generate_report
from ..storage import store

router = APIRouter(tags=["report"])


@router.post(
    "/report",
    response_model=ReportResponse,
    summary="Generate a downloadable PDF evidence report for a completed analysis",
)
def create_report(payload: ReportRequest) -> ReportResponse:
    job = store.get(payload.analysis_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis with that id. Results expire after 24 hours.",
        )
    if job.status is not AnalysisStatus.COMPLETE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Analysis is '{job.status.value}'. A report needs a completed analysis.",
        )

    response = generate_report(job)
    if response.status == "not_implemented":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=response.detail,
        )
    return response
