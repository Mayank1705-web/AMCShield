from fastapi import APIRouter
from ..config import settings
from ..services.dataset_service import DatasetService

router = APIRouter(prefix="/dataset", tags=["Dataset"])
service = DatasetService(settings.data_dir)

@router.get("/metadata")
def metadata():
    return {"success": True, "data": service.overview()}

@router.get("/overview")
def overview():
    return {"success": True, "data": service.overview()}
