"""Structured tracing of graph runs: one span per node, tool call and model call.

``RunTracer`` is a LangChain callback handler, so it sees everything LangGraph and
LangChain already report without touching node code. Internal runnables (edge
functions, sequences) are skipped and their children re-parented to the nearest
recorded span, which leaves a readable tree:

    turn
    ├── guard_input
    ├── router
    │   └── llm
    ├── refunds
    │   └── llm
    └── refunds_tools
        ├── tool:get_order
        └── tool:issue_refund

A finished ``Trace`` serialises to JSON and can be exported to a JSONL file or to
OpenTelemetry (``pip install .[otel]``).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.errors import GraphInterrupt

log = logging.getLogger(__name__)

SpanKind = Literal["graph", "node", "tool", "llm"]
SpanStatus = Literal["ok", "error", "interrupted", "unfinished"]


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    name: str
    kind: SpanKind
    start_ns: int
    end_ns: int | None = None
    status: SpanStatus = "unfinished"
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        return None if self.end_ns is None else round((self.end_ns - self.start_ns) / 1e6, 3)

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "duration_ms": self.duration_ms}


@dataclass
class Trace:
    trace_id: str
    thread_id: str
    operation: str
    spans: list[Span]

    @property
    def root(self) -> Span | None:
        return next((s for s in self.spans if s.parent_id is None), None)

    def to_json(self) -> dict[str, Any]:
        root = self.root
        return {
            "trace_id": self.trace_id,
            "thread_id": self.thread_id,
            "operation": self.operation,
            "duration_ms": root.duration_ms if root else None,
            "spans": [s.to_json() for s in self.spans],
        }


class RunTracer(BaseCallbackHandler):
    def __init__(self) -> None:
        self._spans: dict[UUID, Span] = {}
        # Runs we chose not to record, mapped to the recorded span that adopts their children.
        self._adopted_by: dict[UUID, UUID | None] = {}
        self._lock = threading.Lock()

    # --- span bookkeeping -------------------------------------------------

    def _resolve(self, run_id: UUID | None) -> UUID | None:
        while run_id is not None and run_id not in self._spans:
            run_id = self._adopted_by.get(run_id)
        return run_id

    def _open(
        self, run_id: UUID, parent: UUID | None, name: str, kind: SpanKind, **attrs: Any
    ) -> None:
        with self._lock:
            parent_span = self._resolve(parent)
            self._spans[run_id] = Span(
                span_id=run_id.hex,
                parent_id=self._spans[parent_span].span_id if parent_span else None,
                name=name,
                kind=kind,
                start_ns=time.time_ns(),
                attributes={k: v for k, v in attrs.items() if v is not None},
            )

    def _skip(self, run_id: UUID, parent: UUID | None) -> None:
        with self._lock:
            self._adopted_by[run_id] = parent

    def _close(self, run_id: UUID, error: BaseException | None = None, **attrs: Any) -> None:
        with self._lock:
            span = self._spans.get(run_id)
            if span is None:
                return
            span.end_ns = time.time_ns()
            span.attributes.update({k: v for k, v in attrs.items() if v is not None})
            if error is None:
                span.status = "ok"
            elif isinstance(error, GraphInterrupt):
                span.status = "interrupted"
            else:
                span.status = "error"
                span.error = f"{type(error).__name__}: {error}"

    # --- chains: the graph and its nodes ------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name", "chain")
        node = (metadata or {}).get("langgraph_node")
        if parent_run_id is None:
            self._open(run_id, None, "turn", "graph")
        elif node and name == node:
            self._open(
                run_id, parent_run_id, node, "node", step=(metadata or {}).get("langgraph_step")
            )
        else:
            self._skip(run_id, parent_run_id)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error)

    # --- tools ----------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name", "tool")
        self._open(run_id, parent_run_id, f"tool:{name}", "tool", tool=name)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        result = output if isinstance(output, dict) else {}
        self._close(
            run_id,
            tool_error=result.get("error"),
            retryable=result.get("retryable"),
        )

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error)

    # --- model calls ------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        meta = metadata or {}
        model = meta.get("ls_model_name") or kwargs.get("name") or (serialized or {}).get("name")
        self._open(
            run_id,
            parent_run_id,
            "llm",
            "llm",
            provider=meta.get("ls_provider"),
            model=model,
            input_messages=len(messages[0]) if messages else 0,
        )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        usage: dict[str, Any] = {}
        for generations in getattr(response, "generations", []):
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or usage
        self._close(
            run_id,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error)

    # --- result -------------------------------------------------------------------

    def finish(self, thread_id: str, operation: str) -> Trace:
        with self._lock:
            spans = sorted(self._spans.values(), key=lambda s: s.start_ns)
        return Trace(uuid.uuid4().hex, thread_id, operation, spans)


class TraceExporter(Protocol):
    def export(self, trace: Trace) -> None: ...


class JsonlTraceExporter:
    """Appends one JSON document per trace; easy to grep, tail or load into pandas."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def export(self, trace: Trace) -> None:
        line = json.dumps(trace.to_json(), default=str)
        with self._lock, self._path.open("a") as f:
            f.write(line + "\n")


class OpenTelemetryTraceExporter:
    """Replays a finished trace as OpenTelemetry spans with the original timings.

    Uses the globally configured tracer provider unless one is passed, so any OTel
    exporter (OTLP, console, Jaeger...) set up by the host application applies.
    """

    def __init__(self, tracer: Any | None = None) -> None:
        from opentelemetry import trace as otel_trace

        self._otel = otel_trace
        self._tracer = tracer or otel_trace.get_tracer("pulso_support")

    def export(self, trace: Trace) -> None:
        from opentelemetry.trace import Status, StatusCode

        created: dict[str, Any] = {}
        for span in trace.spans:
            parent = created.get(span.parent_id) if span.parent_id else None
            context = self._otel.set_span_in_context(parent) if parent is not None else None
            attributes = {
                "pulso.trace_id": trace.trace_id,
                "pulso.thread_id": trace.thread_id,
                "pulso.operation": trace.operation,
                "pulso.span_kind": span.kind,
                "pulso.status": span.status,
                **{
                    f"pulso.{k}": v
                    for k, v in span.attributes.items()
                    if isinstance(v, str | int | float | bool)
                },
            }
            otel_span = self._tracer.start_span(
                span.name, context=context, start_time=span.start_ns, attributes=attributes
            )
            if span.status == "error":
                otel_span.set_status(Status(StatusCode.ERROR, span.error))
            otel_span.end(end_time=span.end_ns or span.start_ns)
            created[span.span_id] = otel_span


def export_safely(exporters: list[TraceExporter], trace: Trace) -> None:
    """Tracing must never break a support conversation."""
    for exporter in exporters:
        try:
            exporter.export(trace)
        except Exception:
            log.exception("trace exporter %s failed", type(exporter).__name__)
