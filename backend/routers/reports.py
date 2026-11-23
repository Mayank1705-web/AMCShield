from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..dependencies import require_admin, require_authenticated
from ..services.security_service import require_csrf
from ..config import settings
from ..services.results_service import ResultsService

router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
    dependencies=[Depends(require_authenticated)],
)
service = ResultsService(settings.results_dir)


class ReportRequest(BaseModel):
    report_type: Literal["executive", "metrics", "full"]
    scope: Literal["all", "baseline", "robust"] = "all"
    format: Literal["json", "markdown", "html"] = "json"
    version: Literal["latest"] = "latest"


def _data(builder, *args):
    try:
        return {"success": True, "data": builder(*args)}
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="Evaluation results are unavailable.",
        ) from error


@router.get("/summary")
def summary():
    return _data(service.reports_summary)


@router.get("/artifacts")
def artifacts():
    return {"success": True, "data": service.summary()}


@router.get("/executive-summary")
def executive_summary():
    return _data(service.reports_executive_summary)


@router.get("/metrics")
def metrics():
    return _data(service.reports_metrics)


@router.get("/methodology")
def methodology():
    return {"success": True, "data": service.reports_methodology()}


@router.get("/checks")
def checks():
    return {"success": True, "data": service.reports_checks()}


@router.get("/findings")
def findings():
    return _data(service.evaluation_insights)


@router.get("/options")
def options():
    return {"success": True, "data": service.reports_options()}


@router.get("/export")
def export(format: str = Query("json")):
    if format != "json":
        raise HTTPException(
            status_code=400,
            detail="Unsupported export format. Only 'json' is available.",
        )
    return _data(service.reports_export)


@router.post(
    "/generate",
    dependencies=[Depends(require_csrf), Depends(require_admin)],
)
def generate(payload: ReportRequest):
    fmt = "markdown" if payload.format == "markdown" else "json"
    return _data(
        service.reports_generate,
        payload.report_type,
        payload.scope,
        fmt,
        payload.version,
    )


@router.post(
    "/preview",
    dependencies=[Depends(require_csrf), Depends(require_admin)],
)
def preview(payload: ReportRequest):
    return _data(service.reports_preview, payload.report_type, payload.scope)
