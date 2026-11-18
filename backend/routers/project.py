from fastapi import APIRouter, Depends
from ..dependencies import require_authenticated
from ..config import settings

router = APIRouter(
    prefix="/project",
    tags=["Project"],
    dependencies=[Depends(require_authenticated)],
)

@router.get("/overview")
def overview():
    return {
        "success": True,
        "data": {
            "name": "AMCShield",
            "version": settings.app_version,
        },
    }
