"""HTTP wrapper around the support graph.

uvicorn pulso_support.api:create_app --factory
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
from collections.abc import Iterator
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pulso_support.config import Settings
from pulso_support.graph import RefundDecision
from pulso_support.ratelimit import RateLimiter
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


log = logging.getLogger(__name__)
_END = object()


def run_detached(events: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Drive the graph run in its own thread and relay its events.

    If the client disconnects, the response stops reading but the turn still runs
    to completion: the checkpoint never stays half-way through a turn and the
    trace is still recorded. A crash becomes an `error` event, not a cut stream.
    """
    relay: queue.Queue[Any] = queue.Queue()

    def pump() -> None:
        try:
            for event in events:
                relay.put(event)
        except Exception:
            log.exception("streamed turn failed")
            relay.put({"event": "error", "data": {"message": "The turn failed; try again."}})
        finally:
            relay.put(_END)

    threading.Thread(target=pump, name="pulso-stream", daemon=True).start()
    while (event := relay.get()) is not _END:
        yield event


def _sse(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"


def create_app(
    service: SupportService | None = None, rate_limiter: RateLimiter | None = None
) -> FastAPI:
    service = service or SupportService.from_settings(Settings.from_env())
    settings = service.settings
    limiter = rate_limiter or RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)
    app = FastAPI(title="Pulso support agents", version="0.1.0")

    def per_thread_limit(thread_id: str) -> None:
        """Messages and streams on one thread share a token bucket."""
        retry_after = limiter.acquire(thread_id)
        if retry_after is not None:
            raise HTTPException(
                429,
                "Too many messages on this thread; slow down.",
                headers={"Retry-After": str(math.ceil(retry_after))},
            )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/threads/{thread_id}/messages",
        response_model=TurnOut,
        dependencies=[Depends(per_thread_limit)],
    )
    def post_message(thread_id: str, body: MessageIn) -> TurnOut:
        try:
            return TurnOut.of(service.send(thread_id, body.message))
        except ApprovalPendingError:
            raise HTTPException(
                409, "Thread is waiting for a refund approval; POST /resume first."
            ) from None

    @app.post("/threads/{thread_id}/messages/stream", dependencies=[Depends(per_thread_limit)])
    def stream_message(thread_id: str, body: MessageIn) -> StreamingResponse:
        """Server-sent events: route, token, tool_call, then done (or error)."""
        try:
            events = service.stream(thread_id, body.message)
        except ApprovalPendingError:
            raise HTTPException(
                409, "Thread is waiting for a refund approval; POST /resume first."
            ) from None
        return StreamingResponse(
            _sse(run_detached(events)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/threads/{thread_id}/resume", response_model=TurnOut)
    def resume(thread_id: str, decision: RefundDecision) -> TurnOut:
        try:
            return TurnOut.of(service.resume(thread_id, decision))
        except NoPendingApprovalError:
            raise HTTPException(409, "Thread has no pending approval.") from None

    @app.get("/threads/{thread_id}/traces")
    def get_traces(thread_id: str) -> dict[str, Any]:
        """Span trees of the most recent runs on this thread, newest last."""
        return {
            "thread_id": thread_id,
            "traces": [t.to_json() for t in service.traces(thread_id)],
        }

    @app.get("/threads/{thread_id}", response_model=HistoryOut)
    def get_thread(thread_id: str) -> HistoryOut:
        return HistoryOut(
            thread_id=thread_id,
            messages=service.history(thread_id),
            summary=service.summary(thread_id),
            pending_approval=service.pending_approval(thread_id),
        )

    return app
