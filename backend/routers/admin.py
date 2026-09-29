from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..config import settings


router = APIRouter(prefix="/admin", tags=["Admin"])


class MessageStatusUpdate(BaseModel):
    status: str = Field(
        ...,
        pattern="^(new|read|replied|closed)$",
    )


def _messages_file() -> Path:
    return settings.results_dir / "admin_messages.json"


def _require_admin(request: Request) -> dict[str, Any]:
    user = request.session.get("user")

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
        )

    role = str(user.get("role", "")).strip().lower()

    if role != "administrator":
        raise HTTPException(
            status_code=403,
            detail="Administrator access required.",
        )

    return user


def _read_messages() -> list[dict[str, Any]]:
    path = _messages_file()

    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        return data

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to read administrator messages.",
        ) from exc


def _write_messages(messages: list[dict[str, Any]]) -> None:
    path = _messages_file()
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp")

    try:
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)

        temp_path.replace(path)

    except OSError as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Unable to save administrator messages.",
        ) from exc


@router.get("/messages")
def list_messages(
    request: Request,
    _: dict[str, Any] = Depends(_require_admin),
):
    messages = _read_messages()

    messages = sorted(
        messages,
        key=lambda item: str(item.get("created_at", "")),
        reverse=True,
    )

    counts = {
        "total": len(messages),
        "new": sum(
            1 for item in messages
            if item.get("status") == "new"
        ),
        "read": sum(
            1 for item in messages
            if item.get("status") == "read"
        ),
        "replied": sum(
            1 for item in messages
            if item.get("status") == "replied"
        ),
        "closed": sum(
            1 for item in messages
            if item.get("status") == "closed"
        ),
    }

    return {
        "success": True,
        "messages": messages,
        "counts": counts,
    }


@router.get("/messages/{message_id}")
def get_message(
    message_id: int,
    request: Request,
    _: dict[str, Any] = Depends(_require_admin),
):
    messages = _read_messages()

    message = next(
        (
            item
            for item in messages
            if int(item.get("id", -1)) == message_id
        ),
        None,
    )

    if message is None:
        raise HTTPException(
            status_code=404,
            detail="Message not found.",
        )

    return {
        "success": True,
        "message": message,
    }


@router.patch("/messages/{message_id}")
def update_message_status(
    message_id: int,
    payload: MessageStatusUpdate,
    request: Request,
    _: dict[str, Any] = Depends(_require_admin),
):
    messages = _read_messages()

    for message in messages:
        if int(message.get("id", -1)) == message_id:
            message["status"] = payload.status
            _write_messages(messages)

            return {
                "success": True,
                "message": message,
            }

    raise HTTPException(
        status_code=404,
        detail="Message not found.",
    )