from fastapi import APIRouter, Depends
from ..dependencies import require_authenticated
from ..config import settings
from ..services.results_service import ResultsService

router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"],
    dependencies=[Depends(require_authenticated)],
)

service = ResultsService(settings.results_dir)


@router.get("/summary")
def summary():
    return {
        "success": True,
        "data": service.evaluation_summary(),
    }


@router.get("/artifacts")
def artifacts():
    return {
        "success": True,
        "data": service.summary(),
    }


@router.get("/filters")
def filters():
    return {
        "success": True,
        "data": service.evaluation_filters(),
    }


@router.get("/comparison")
def comparison():
    return {
        "success": True,
        "data": service.evaluation_comparison(),
    }


@router.get("/attack-summary")
def attack_summary():
    return {
        "success": True,
        "data": service.evaluation_attack_summary(),
    }


@router.get("/results")
def results(
    attack: str | None = None,
    snr: int | None = None,
    model: str | None = None,
):
    return {
        "success": True,
        "data": service.evaluation_results(
            attack=attack,
            snr=snr,
            model=model,
        ),
    }


@router.get("/accuracy-by-snr")
def accuracy_by_snr():
    return {
        "success": True,
        "data": service.evaluation_accuracy_by_snr(),
    }


@router.get("/generalization-gap")
def generalization_gap():
    return {
        "success": True,
        "data": service.evaluation_generalization_gap(),
    }


@router.get("/insights")
def insights():
    return {
        "success": True,
        "data": service.evaluation_insights(),
    }
