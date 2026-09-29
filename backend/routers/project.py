from fastapi import APIRouter
from ..config import settings, PROJECT_ROOT

router = APIRouter(prefix="/project", tags=["Project"])

@router.get("/overview")
def overview():
    return {
        "success": True,
        "data": {
            "name": "AMCShield",
            "version": settings.app_version,
            "project_root": str(PROJECT_ROOT),
        },
    }
