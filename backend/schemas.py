from typing import Any
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

class APIResponse(BaseModel):
    success: bool = True
    data: Any = None
    message: str | None = None
