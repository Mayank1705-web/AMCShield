from fastapi import APIRouter
from ..config import settings
from ..services.training_service import TrainingService

router = APIRouter(prefix="/training", tags=["Training"])
service = TrainingService(settings.checkpoints_dir, settings.src_dir)

@router.get("/overview")
def overview():
    return {"success": True, "data": service.status()}

@router.get("/status")
def status():
    return {"success": True, "data": service.status()}

@router.get("/checkpoints")
def checkpoints():
    return {"success": True, "data": service.status()["checkpoints"]}
