from fastapi import APIRouter, Depends, HTTPException, Query

from ..config import settings
from ..dependencies import require_authenticated
from ..services.dataset_service import DatasetService


router = APIRouter(
    prefix="/dataset",
    tags=["Dataset"],
    dependencies=[Depends(require_authenticated)],
)

service = DatasetService(settings.data_dir)


def execute_service(method, *args):
    try:
        return {
            "success": True,
            "data": method(*args),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/metadata")
def metadata():
    return execute_service(service.overview)


@router.get("/overview")
def overview():
    return execute_service(service.overview)


@router.get("/class-distribution")
def class_distribution():
    return execute_service(service.class_distribution)


@router.get("/snr-distribution")
def snr_distribution():
    return execute_service(service.snr_distribution)


@router.get("/splits")
def splits():
    return execute_service(service.splits)


@router.get("/quality")
def quality():
    return execute_service(service.quality)


@router.get("/sample")
def sample(
    class_name: str = Query(..., min_length=1),
    snr: int = Query(...),
):
    return execute_service(service.sample, class_name, snr)
