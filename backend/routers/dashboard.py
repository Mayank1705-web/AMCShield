from fastapi import APIRouter

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/overview")
def overview():
    return {"success": True, "data": {"service": "dashboard", "status": "ready"}}
