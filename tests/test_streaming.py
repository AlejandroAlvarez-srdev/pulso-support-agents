from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from pulso_support import fixtures as fx
from pulso_support.api import create_app
from pulso_support.guardrails import StreamingRedactor
from pulso_support.llm import ScriptedSupportModel


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_scripted_model_streams_text_word_by_word():
    model = ScriptedSupportModel(tool_names=("get_order", "get_assignment"))
    chunks = [c.text for c in model.stream("where is my order?") if c.text]

    assert chunks[:3] == ["Could", " you", " share"]
    assert "".join(chunks) == model.invoke("where is my order?").text


def test_scripted_model_streams_tool_calls_as_one_chunk():
    model = ScriptedSupportModel(tool_names=("get_order", "get_assignment"))
    chunks = list(model.stream(f"where is {fx.ORDER_DELIVERED}?"))
    calls = [c for c in chunks if c.tool_call_chunks]

    assert len(calls) == 1
    assert [t["name"] for t in calls[0].tool_calls] == ["get_order", "get_assignment"]


def test_stream_emits_route_tool_calls_tokens_and_done(make_service):
    events = list(make_service().stream("s-1", f"Where is order {fx.ORDER_IN_PROGRESS}?"))
    kinds = [e["event"] for e in events]

    assert kinds[0] == "route" and events[0]["data"] == {"route": "order_status"}
    assert [e["data"]["name"] for e in events if e["event"] == "tool_call"] == [
        "get_order",
        "get_assignment",
    ]
    assert kinds.count("token") > 5
    assert kinds[-1] == "done"
    streamed = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    assert streamed == events[-1]["data"]["reply"]


def test_stream_ends_with_pending_approval_for_large_refunds(make_service):
    events = list(make_service().stream("s-2", f"refund order {fx.ORDER_DELIVERED_LARGE}"))
    done = events[-1]["data"]

    assert done["status"] == "awaiting_approval"
    assert done["pending_approval"]["amount_cents"] == 189_000
    assert not any(e["event"] == "token" for e in events)


def test_streamed_tokens_never_reveal_pii(make_service):
    class LeakyModel(ScriptedSupportModel):
        def _act(self, messages):
            message = super()._act(messages)
            if message.tool_calls:
                return message
            return AIMessage(message.text + " Call +52 55 1234 5678 or write to ops@pulso.mx.")

    events = list(make_service(model=LeakyModel()).stream("s-3", f"where is {fx.ORDER_DELIVERED}"))
    streamed = "".join(e["data"]["text"] for e in events if e["event"] == "token")

    assert "1234" not in streamed and "ops@pulso.mx" not in streamed
    assert streamed.endswith("Call [phone redacted] or write to [email redacted].")
    assert streamed == events[-1]["data"]["reply"]


@pytest.mark.parametrize(
    "tokens",
    [
        ["Call +52 ", "55 12", "34 5678", " now"],
        ["mail ana", "@exam", "ple.com", " ok"],
        ["ETA ", "18", " min"],
    ],
)
def test_streaming_redactor_matches_batch_redaction(tokens):
    from pulso_support.guardrails import redact_pii

    redactor = StreamingRedactor()
    out = "".join(redactor.feed(t) for t in tokens) + redactor.flush()
    assert out == redact_pii("".join(tokens))


def test_sse_endpoint(make_service):
    client = TestClient(create_app(make_service()))

    with client.stream(
        "POST",
        "/threads/s-4/messages/stream",
        json={"message": f"Where is order {fx.ORDER_IN_PROGRESS}?"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(response.read().decode())

    assert events[0] == ("route", {"route": "order_status"})
    assert events[-1][0] == "done"
    assert "Ana Torres" in events[-1][1]["reply"]


def test_sse_endpoint_rejects_paused_threads(make_service):
    client = TestClient(create_app(make_service()))
    client.post("/threads/s-5/messages", json={"message": f"refund {fx.ORDER_DELIVERED_LARGE}"})

    response = client.post("/threads/s-5/messages/stream", json={"message": "hi"})
    assert response.status_code == 409
