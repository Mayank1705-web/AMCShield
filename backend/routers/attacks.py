from fastapi import APIRouter
from ..config import settings
from ..services.results_service import ResultsService

router = APIRouter(prefix="/attacks", tags=["Adversarial Attacks"])
service = ResultsService(settings.results_dir)

@router.get("/overview")
def overview():
    return {"success": True, "data": service.summary()}

@router.get("/results")
def results():
    return {"success": True, "data": service.summary()}
