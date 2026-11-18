
from fastapi import APIRouter, Depends

from ..dependencies import require_authenticated
from ..config import settings
from ..services.results_service import ResultsService


router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"],
    dependencies=[Depends(require_authenticated)],
)

service = ResultsService(settings.results_dir)


@router.get("/summary")
def summary():
    return {
        "success": True,
        "data": service.dashboard_summary(),
    }


@router.get("/performance")
def performance():
    return {
        "success": True,
        "data": service.dashboard_performance(),
    }


@router.get("/attacks")
def attacks():
    return {
        "success": True,
        "data": service.dashboard_attacks(),
    }


@router.get("/generalization-gap")
def generalization_gap():
    return {
        "success": True,
        "data": service.dashboard_generalization_gap(),
    }


@router.get("/snr")
def snr():
    return {
        "success": True,
        "data": service.dashboard_snr(),
    }


@router.get("/recent-updates")
def recent_updates():
    return {
        "success": True,
        "data": service.dashboard_recent_updates(),
    }
