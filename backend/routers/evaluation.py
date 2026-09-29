from fastapi import APIRouter
from ..config import settings
from ..services.results_service import ResultsService

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])
service = ResultsService(settings.results_dir)

@router.get("/summary")
def summary():
    return {"success": True, "data": service.summary()}

@router.get("/artifacts")
def artifacts():
    return {"success": True, "data": service.summary()}
