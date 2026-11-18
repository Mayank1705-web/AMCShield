from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..dependencies import require_admin, require_authenticated
from ..services.security_service import require_csrf
from ..config import settings
from ..services.training_service import TrainingService


router = APIRouter(
    prefix="/training",
    tags=["Training"],
    dependencies=[
        Depends(require_authenticated),
        Depends(require_admin),
    ],
)

service = TrainingService(settings.checkpoints_dir, settings.src_dir)


class TrainingRequest(BaseModel):
    model: Literal["baseline", "robust"] = "baseline"


def _run_action(action, *args, **kwargs):
    try:
        return {"success": True, "data": action(*args, **kwargs)}
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/overview")
def overview():
    return {"success": True, "data": service.overview()}


@router.get("/status")
def status():
    return {"success": True, "data": service.status()}


@router.get("/config")
def config():
    try:
        return {"success": True, "data": service.config()}
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@router.get("/progress")
def progress():
    return {"success": True, "data": service.progress()}


@router.get("/history")
def history():
    return {"success": True, "data": service.history()}


@router.get("/models")
def models():
    return {"success": True, "data": service.models()}


@router.get("/checkpoints")
def checkpoints():
    return {"success": True, "data": service.checkpoints()}


@router.get("/logs")
def logs():
    return {"success": True, "data": service.logs()}


@router.post("/start", dependencies=[Depends(require_csrf)])
def start(payload: TrainingRequest | None = None):
    request = payload or TrainingRequest()
    return _run_action(service.start, request.model)


@router.post("/resume", dependencies=[Depends(require_csrf)])
def resume(payload: TrainingRequest | None = None):
    request = payload or TrainingRequest()
    return _run_action(service.start, request.model, resume_only=True)


@router.post("/stop", dependencies=[Depends(require_csrf)])
def stop():
    return _run_action(service.stop)
