from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import require_authenticated
from ..config import settings
from ..services.results_service import ResultsService

router = APIRouter(
    prefix="/visualization",
    tags=["Visualization"],
    dependencies=[Depends(require_authenticated)],
)
service = ResultsService(settings.results_dir)


def _data(builder):
    try:
        return {"success": True, "data": builder()}
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="Evaluation results are unavailable.",
        ) from error


@router.get("/summary")
def summary():
    return _data(service.visualization_summary)


@router.get("/artifacts")
def artifacts():
    return {"success": True, "data": service.summary()}


@router.get("/filters")
def filters():
    return _data(service.visualization_filters)


@router.get("/accuracy-by-snr")
def accuracy_by_snr():
    return _data(service.evaluation_accuracy_by_snr)


@router.get("/asr-by-snr")
def asr_by_snr():
    return _data(service.dashboard_attacks)


@router.get("/generalization-gap")
def generalization_gap():
    return _data(service.evaluation_generalization_gap)


@router.get("/class-accuracy")
def class_accuracy():
    return _data(service.visualization_class_accuracy)


@router.get("/model-comparison")
def model_comparison():
    return _data(service.visualization_model_comparison)


@router.get("/insights")
def insights():
    return _data(service.evaluation_insights)
