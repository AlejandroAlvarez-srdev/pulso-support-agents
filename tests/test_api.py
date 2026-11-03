from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pulso_support import fixtures as fx
from pulso_support.api import create_app


@pytest.fixture
def client(make_service) -> TestClient:
    return TestClient(create_app(make_service()))


def test_message_round_trip(client: TestClient):
    response = client.post(
        "/threads/api-1/messages", json={"message": f"Where is order {fx.ORDER_IN_PROGRESS}?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["route"] == "order_status"
    assert [c["name"] for c in body["tool_calls"]] == ["get_order", "get_assignment"]
    assert "Ana Torres" in body["reply"]


def test_refund_approval_flow_over_http(client: TestClient, backend):
    paused = client.post(
        "/threads/api-2/messages", json={"message": f"refund order {fx.ORDER_DELIVERED_LARGE}"}
    ).json()
    assert paused["status"] == "awaiting_approval"
    assert paused["pending_approval"]["amount_cents"] == 189_000

    blocked = client.post("/threads/api-2/messages", json={"message": "hello?"})
    assert blocked.status_code == 409

    thread = client.get("/threads/api-2").json()
    assert thread["pending_approval"]["order_id"] == fx.ORDER_DELIVERED_LARGE

    done = client.post("/threads/api-2/resume", json={"approved": True, "reviewer": "ops"})
    assert done.status_code == 200
    assert done.json()["status"] == "completed"
    assert backend.refunds_for(fx.ORDER_DELIVERED_LARGE)[0].approved_by == "ops"


def test_resume_without_pending_approval_is_a_conflict(client: TestClient):
    response = client.post("/threads/api-3/resume", json={"approved": True})
    assert response.status_code == 409


def test_validation_rejects_empty_messages(client: TestClient):
    assert client.post("/threads/api-4/messages", json={"message": ""}).status_code == 422


def test_healthz(client: TestClient):
    assert client.get("/healthz").json() == {"status": "ok"}
