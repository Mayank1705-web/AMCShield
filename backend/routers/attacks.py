from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import require_admin, require_authenticated
from ..services.security_service import require_csrf
from ..config import settings
from ..services.results_service import ResultsService

router = APIRouter(
    prefix="/attacks",
    tags=["Adversarial Attacks"],
    dependencies=[Depends(require_authenticated)],
)
service = ResultsService(settings.results_dir)

LIVE_ATTACKS_DISABLED = (
    "Live attack execution is disabled; attacks are precomputed offline "
    "(see CONSTRAINTS.md)."
)


def _data(builder):
    try:
        return {"success": True, "data": builder()}
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="Evaluation results are unavailable.",
        ) from error


@router.get("/overview")
def overview():
    return _data(service.attacks_overview)


@router.get("/results")
def results():
    return _data(service.attacks_results)


@router.get("/config")
def config():
    return {"success": True, "data": service.attacks_config()}


@router.get("/snr-profile")
def snr_profile():
    return _data(service.attacks_snr_profile)


@router.get("/methods")
def methods():
    return {"success": True, "data": service.attacks_methods()}


@router.get("/constraints")
def constraints():
    return {"success": True, "data": service.attacks_constraints()}


@router.get("/logs")
def logs():
    return {"success": True, "data": []}


@router.post(
    "/run",
    dependencies=[Depends(require_csrf), Depends(require_admin)],
)
def run():
    raise HTTPException(status_code=501, detail=LIVE_ATTACKS_DISABLED)


@router.post(
    "/stop",
    dependencies=[Depends(require_csrf), Depends(require_admin)],
)
def stop():
    raise HTTPException(status_code=501, detail=LIVE_ATTACKS_DISABLED)
