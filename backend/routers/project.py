from fastapi import APIRouter, Depends

from ..dependencies import require_authenticated
from ..config import settings
from ..services.results_service import ResultsService

router = APIRouter(
    prefix="/project",
    tags=["Project"],
    dependencies=[Depends(require_authenticated)],
)
service = ResultsService(settings.results_dir)


@router.get("/overview")
def overview():
    data = {
        "name": "AMCShield",
        "version": settings.app_version,
    }
    data.update(service.project_overview())
    return {"success": True, "data": data}


@router.get("/mission")
def mission():
    return {"success": True, "data": service.project_mission()}


@router.get("/facts")
def facts():
    return {"success": True, "data": service.project_facts()}


@router.get("/models")
def models():
    return {"success": True, "data": service.project_models()}


@router.get("/attacks")
def attacks():
    return {"success": True, "data": service.project_attacks()}


@router.get("/context")
def context():
    return {"success": True, "data": service.project_context()}


@router.get("/resources")
def resources():
    # No curated external links are stored in the repository; the page keeps
    # its static resource links when this list is empty.
    return {"success": True, "data": []}
