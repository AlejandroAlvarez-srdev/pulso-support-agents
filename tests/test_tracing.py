from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from pulso_support import fixtures as fx
from pulso_support.api import create_app
from pulso_support.graph import RefundDecision
from pulso_support.service import SupportService
from pulso_support.tracing import JsonlTraceExporter, OpenTelemetryTraceExporter, Trace


def _tree(trace: Trace) -> dict[str, list[str]]:
    by_id = {s.span_id: s for s in trace.spans}
    children: dict[str, list[str]] = {}
    for span in trace.spans:
        parent = by_id[span.parent_id].name if span.parent_id else "-"
        children.setdefault(parent, []).append(span.name)
    return children


def test_trace_has_a_span_per_node_tool_and_model_call(make_service):
    service = make_service()
    service.send("tr-1", f"Where is order {fx.ORDER_IN_PROGRESS}?")

    [trace] = service.traces("tr-1")
    tree = _tree(trace)

    assert tree["-"] == ["turn"]
    assert tree["turn"] == [
        "guard_input",
        "compact",
        "router",
        "order_status",
        "order_status_tools",
        "order_status",
        "guard_output",
    ]
    assert tree["router"] == ["llm"]
    assert tree["order_status_tools"] == ["tool:get_order", "tool:get_assignment"]
    assert all(s.status == "ok" and s.duration_ms >= 0 for s in trace.spans)


def test_interrupted_runs_are_marked_and_resume_gets_its_own_trace(make_service):
    service = make_service()
    service.send("tr-2", f"refund order {fx.ORDER_DELIVERED_LARGE}")
    service.resume("tr-2", RefundDecision(approved=True))

    paused, resumed = service.traces("tr-2")
    assert [s.status for s in paused.spans if s.name == "refunds_tools"] == ["ok", "interrupted"]
    assert resumed.operation == "resume"
    refund = next(s for s in resumed.spans if s.name == "tool:issue_refund")
    assert refund.status == "ok"


def test_tool_errors_are_recorded_as_attributes(make_service):
    service = make_service()
    service.send("tr-3", "where is order 0b5e1c7a-2f43-4c8e-9a1d-6f2e3b4c5dff?")

    [trace] = service.traces("tr-3")
    get_order = next(s for s in trace.spans if s.name == "tool:get_order")
    assert "does not exist" in get_order.attributes["tool_error"]


def test_jsonl_exporter_writes_one_line_per_run(make_service, tmp_path):
    path = tmp_path / "traces.jsonl"
    service = make_service()
    service.trace_exporters.append(JsonlTraceExporter(path))

    service.send("tr-4", f"Where is order {fx.ORDER_IN_PROGRESS}?")
    service.send("tr-4", "thanks!")

    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["thread_id"] for line in lines] == ["tr-4", "tr-4"]
    assert lines[0]["spans"][0]["name"] == "turn"
    assert lines[0]["duration_ms"] > 0


def test_a_failing_exporter_does_not_break_the_conversation(make_service):
    class Broken:
        def export(self, trace):
            raise OSError("disk full")

    service: SupportService = make_service()
    service.trace_exporters.append(Broken())

    assert service.send("tr-5", "talk to a human").ticket_id is not None


def test_opentelemetry_export_preserves_hierarchy_and_timing(make_service):
    sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
    in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    memory = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    service = make_service()
    service.trace_exporters.append(OpenTelemetryTraceExporter(provider.get_tracer("test")))

    service.send("tr-6", f"Where is order {fx.ORDER_IN_PROGRESS}?")

    spans = {s.name: s for s in memory.get_finished_spans()}
    [ours] = service.traces("tr-6")
    assert len(memory.get_finished_spans()) == len(ours.spans)
    root, tool = spans["turn"], spans["tool:get_order"]
    assert root.parent is None
    assert tool.context.trace_id == root.context.trace_id
    assert spans["order_status_tools"].context.span_id == tool.parent.span_id
    assert tool.attributes["pulso.thread_id"] == "tr-6"
    assert tool.start_time == next(s.start_ns for s in ours.spans if s.name == "tool:get_order")


def test_traces_endpoint(make_service):
    client = TestClient(create_app(make_service()))
    client.post(
        "/threads/tr-7/messages/stream",
        json={"message": f"Where is order {fx.ORDER_IN_PROGRESS}?"},
    )

    body = client.get("/threads/tr-7/traces").json()
    [trace] = body["traces"]
    assert trace["operation"] == "stream"
    assert {"tool:get_order", "tool:get_assignment"} <= {s["name"] for s in trace["spans"]}
