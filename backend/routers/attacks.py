from fastapi import APIRouter, Depends
from ..dependencies import require_authenticated
from ..config import settings
from ..services.results_service import ResultsService

router = APIRouter(
    prefix="/attacks",
    tags=["Adversarial Attacks"],
    dependencies=[Depends(require_authenticated)],
)
service = ResultsService(settings.results_dir)

@router.get("/overview")
def overview():
    return {"success": True, "data": service.summary()}

@router.get("/results")
def results():
    return {"success": True, "data": service.summary()}
