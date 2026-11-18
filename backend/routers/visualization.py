from fastapi import APIRouter, Depends
from ..dependencies import require_authenticated
from ..config import settings
from ..services.results_service import ResultsService

router = APIRouter(
    prefix="/visualization",
    tags=["Visualization"],
    dependencies=[Depends(require_authenticated)],
)
service = ResultsService(settings.results_dir)

@router.get("/summary")
def summary():
    return {"success": True, "data": service.summary()}

@router.get("/artifacts")
def artifacts():
    return {"success": True, "data": service.summary()}
