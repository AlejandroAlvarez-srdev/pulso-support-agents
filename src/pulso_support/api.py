"""HTTP wrapper around the support graph.

uvicorn pulso_support.api:create_app --factory
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pulso_support.config import Settings
from pulso_support.graph import RefundDecision
from pulso_support.service import (
    ApprovalPendingError,
    NoPendingApprovalError,
    SupportService,
    TurnResult,
)


class MessageIn(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)


class TurnOut(BaseModel):
    thread_id: str
    status: Literal["completed", "awaiting_approval"]
    route: str | None
    reply: str | None
    tool_calls: list[dict[str, Any]]
    pending_approval: dict[str, Any] | None
    ticket_id: str | None

    @classmethod
    def of(cls, result: TurnResult) -> TurnOut:
        return cls.model_validate(result, from_attributes=True)


class HistoryOut(BaseModel):
    thread_id: str
    messages: list[dict[str, str]]
    summary: str | None
    pending_approval: dict[str, Any] | None


def _sse(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"


def create_app(service: SupportService | None = None) -> FastAPI:
    service = service or SupportService.from_settings(Settings.from_env())
    app = FastAPI(title="Pulso support agents", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/threads/{thread_id}/messages", response_model=TurnOut)
    def post_message(thread_id: str, body: MessageIn) -> TurnOut:
        try:
            return TurnOut.of(service.send(thread_id, body.message))
        except ApprovalPendingError:
            raise HTTPException(
                409, "Thread is waiting for a refund approval; POST /resume first."
            ) from None

    @app.post("/threads/{thread_id}/messages/stream")
    def stream_message(thread_id: str, body: MessageIn) -> StreamingResponse:
        """Server-sent events: route, token, tool_call, then done."""
        try:
            events = service.stream(thread_id, body.message)
        except ApprovalPendingError:
            raise HTTPException(
                409, "Thread is waiting for a refund approval; POST /resume first."
            ) from None
        return StreamingResponse(
            _sse(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/threads/{thread_id}/resume", response_model=TurnOut)
    def resume(thread_id: str, decision: RefundDecision) -> TurnOut:
        try:
            return TurnOut.of(service.resume(thread_id, decision))
        except NoPendingApprovalError:
            raise HTTPException(409, "Thread has no pending approval.") from None

    @app.get("/threads/{thread_id}", response_model=HistoryOut)
    def get_thread(thread_id: str) -> HistoryOut:
        return HistoryOut(
            thread_id=thread_id,
            messages=service.history(thread_id),
            summary=service.summary(thread_id),
            pending_approval=service.pending_approval(thread_id),
        )

    return app
