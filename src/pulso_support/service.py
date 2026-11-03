"""Thread-oriented facade over the compiled graph, shared by the API, evals and demo."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from pulso_support.backend import FakePulsoBackend
from pulso_support.config import Settings
from pulso_support.graph import RefundDecision, build_support_graph, recursion_limit
from pulso_support.ledger import SqliteRefundLedger
from pulso_support.llm import create_chat_model


class ApprovalPendingError(Exception):
    """The thread is paused on a refund approval and cannot take new messages."""


class NoPendingApprovalError(Exception):
    """A resume was requested for a thread that is not paused."""


@dataclass
class TurnResult:
    thread_id: str
    status: Literal["completed", "awaiting_approval"]
    route: str | None
    reply: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pending_approval: dict[str, Any] | None = None
    ticket_id: str | None = None


def create_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    if settings.checkpoint_db:
        conn = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
        return SqliteSaver(conn)
    return InMemorySaver()


def turn_tool_calls(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Tool calls the agents requested since the latest user message."""
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    return [
        {"name": call["name"], "args": call["args"]}
        for m in messages[start:]
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]


class SupportService:
    def __init__(self, graph: CompiledStateGraph, settings: Settings | None = None) -> None:
        self.graph = graph
        self.settings = settings or Settings()

    @classmethod
    def from_settings(
        cls, settings: Settings, backend: FakePulsoBackend | None = None
    ) -> SupportService:
        if backend is None:
            ledger = SqliteRefundLedger(settings.backend_db) if settings.backend_db else None
            backend = FakePulsoBackend.with_fixtures(ledger=ledger)
        graph = build_support_graph(
            create_chat_model(settings),
            backend,
            settings,
            checkpointer=create_checkpointer(settings),
        )
        return cls(graph, settings)

    def _config(self, thread_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit(self.settings),
        }

    def pending_approval(self, thread_id: str) -> dict[str, Any] | None:
        interrupts = self.graph.get_state(self._config(thread_id)).interrupts
        return dict(interrupts[0].value) if interrupts else None

    def send(self, thread_id: str, text: str) -> TurnResult:
        if self.pending_approval(thread_id):
            raise ApprovalPendingError(thread_id)
        self.graph.invoke({"messages": [HumanMessage(text)]}, self._config(thread_id))
        return self._result(thread_id)

    def resume(self, thread_id: str, decision: RefundDecision) -> TurnResult:
        if not self.pending_approval(thread_id):
            raise NoPendingApprovalError(thread_id)
        self.graph.invoke(Command(resume=decision.model_dump()), self._config(thread_id))
        return self._result(thread_id)

    def history(self, thread_id: str) -> list[dict[str, str]]:
        values = self.graph.get_state(self._config(thread_id)).values
        return [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.text}
            for m in values.get("messages", [])
            if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
        ]

    def summary(self, thread_id: str) -> str | None:
        return self.graph.get_state(self._config(thread_id)).values.get("summary")

    def _result(self, thread_id: str) -> TurnResult:
        snapshot = self.graph.get_state(self._config(thread_id))
        values = snapshot.values
        messages = values.get("messages", [])
        pending = dict(snapshot.interrupts[0].value) if snapshot.interrupts else None
        last = messages[-1] if messages else None
        reply = last.text if isinstance(last, AIMessage) and not last.tool_calls else None
        return TurnResult(
            thread_id=thread_id,
            status="awaiting_approval" if pending else "completed",
            route=values.get("route"),
            reply=None if pending else reply,
            tool_calls=turn_tool_calls(messages),
            pending_approval=pending,
            ticket_id=values.get("ticket_id"),
        )
