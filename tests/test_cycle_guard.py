from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from pulso_support import fixtures as fx
from pulso_support.config import Settings
from pulso_support.llm import ScriptedSupportModel


class StuckModel(ScriptedSupportModel):
    """Routes like the scripted model but, as an agent, asks for get_order forever."""

    def _act(self, messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_order",
                    "args": {"order_id": fx.ORDER_IN_PROGRESS},
                    "id": f"loop_{len(messages)}",
                    "type": "tool_call",
                }
            ],
        )


def test_looping_agent_is_escalated_after_max_steps(make_service, backend):
    service = make_service(model=StuckModel(), settings=Settings(max_agent_steps=3))

    result = service.send("loop-1", f"Where is order {fx.ORDER_IN_PROGRESS}?")

    assert result.route == "order_status"
    assert result.ticket_id is not None
    assert "specialist" in result.reply
    assert len(result.tool_calls) == 3
    assert "3-step limit" in backend.tickets[result.ticket_id]["summary"]


def test_escalation_closes_dangling_tool_calls(make_service):
    service = make_service(model=StuckModel(), settings=Settings(max_agent_steps=2))
    service.send("loop-2", f"Where is order {fx.ORDER_IN_PROGRESS}?")

    messages = service.graph.get_state(service._config("loop-2")).values["messages"]
    requested = {c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    assert requested == answered


def test_step_budget_resets_on_each_turn(make_service):
    service = make_service(settings=Settings(max_agent_steps=2))
    for _ in range(3):
        result = service.send("loop-3", f"Where is order {fx.ORDER_IN_PROGRESS}?")
        assert result.ticket_id is None
        assert "Ana Torres" in result.reply
