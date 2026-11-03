"""The support graph: a router that dispatches to specialised tool-using agents.

    START -> router -> {order_status | refunds | pulser_ops | escalation}
    agent -> agent_tools -> agent      (tool loop, bounded by the cycle guard)
    agent -> END                       (final answer)
    agent -> escalation -> END         (step budget exhausted)

Refunds above the approval threshold pause inside ``refunds_tools`` with a
LangGraph ``interrupt``; the run resumes with ``Command(resume=decision)``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from pulso_support import prompts
from pulso_support.backend import FakePulsoBackend
from pulso_support.config import Settings
from pulso_support.resilience import ResilientExecutor, RetryPolicy
from pulso_support.tools import (
    GET_ASSIGNMENT,
    GET_ORDER,
    INJECTED_ARGS,
    ISSUE_REFUND,
    REASSIGN_ORDER,
    build_tools,
    default_policies,
)

Route = Literal["order_status", "refunds", "pulser_ops", "escalation"]

AGENT_TOOLS: dict[str, tuple[str, ...]] = {
    "order_status": (GET_ORDER, GET_ASSIGNMENT),
    "refunds": (GET_ORDER, ISSUE_REFUND),
    "pulser_ops": (REASSIGN_ORDER,),
}


class SupportState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    route: Route
    # Agent invocations in the current user turn; reset by the router.
    steps: int
    ticket_id: str | None


class RefundDecision(BaseModel):
    """Payload a human reviewer sends to resume a paused refund."""

    approved: bool
    reviewer: str = "support-reviewer"
    note: str | None = None


def parse_route(text: str) -> Route:
    """Map free-form router output to a route; anything unrecognised escalates."""
    cleaned = text.strip().lower()
    for route in prompts.ROUTES:
        if cleaned.startswith(route) or cleaned.strip("`'\". ") == route:
            return route  # type: ignore[return-value]
    for route in prompts.ROUTES:
        if route in cleaned:
            return route  # type: ignore[return-value]
    return "escalation"


def agent_view(messages: list[BaseMessage]) -> list[BaseMessage]:
    """What an agent sees: the transcript of past turns plus the full current turn.

    Tool traces from earlier turns (possibly produced by other agents with other
    tool sets) are dropped; providers reject tool calls that reference tools the
    current request did not declare.
    """
    human_idx = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not human_idx:
        return list(messages)
    start = human_idx[-1]
    past = [
        m
        for m in messages[:start]
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
    ]
    return past + list(messages[start:])


def _make_router(model: BaseChatModel) -> Callable[[SupportState], dict]:
    def router(state: SupportState) -> dict:
        last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        reply = model.invoke([SystemMessage(prompts.ROUTER), HumanMessage(last_human.text)])
        return {"route": parse_route(reply.text), "steps": 0, "ticket_id": None}

    return router


def _make_agent(
    name: str, model: BaseChatModel, tools: list[BaseTool]
) -> Callable[[SupportState], dict]:
    bound = model.bind_tools(tools)
    system = SystemMessage(prompts.AGENT_PROMPTS[name])

    def agent(state: SupportState) -> dict:
        reply = bound.invoke([system, *agent_view(state["messages"])])
        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    agent.__name__ = name
    return agent


def refund_key(thread_id: str, tool_call_id: str) -> str:
    """Stable across re-executions: both parts are stored in the checkpoint."""
    return f"{thread_id}:{tool_call_id}"


def _make_tool_node(tools: dict[str, BaseTool], threshold_cents: int) -> Callable[..., dict]:
    def run_tools(state: SupportState, config: RunnableConfig) -> dict:
        calls = state["messages"][-1].tool_calls
        thread_id = config.get("configurable", {}).get("thread_id", "")

        # Collect every approval before any side effect: on resume LangGraph re-runs
        # this node from the top, so nothing above this line may mutate the backend.
        decisions: dict[str, RefundDecision] = {}
        for call in calls:
            amount = call["args"].get("amount_cents", 0)
            if call["name"] == ISSUE_REFUND and amount > threshold_cents:
                raw = interrupt(
                    {
                        "type": "refund_approval",
                        "tool_call_id": call["id"],
                        "order_id": call["args"].get("order_id"),
                        "amount_cents": amount,
                        "reason": call["args"].get("reason"),
                        "threshold_cents": threshold_cents,
                    }
                )
                decisions[call["id"]] = RefundDecision.model_validate(raw)

        results: list[ToolMessage] = []
        for call in calls:
            tool = tools.get(call["name"])
            if tool is None:
                output: Any = {"error": f"Tool {call['name']} is not available to this agent."}
            elif (decision := decisions.get(call["id"])) and not decision.approved:
                output = {"status": "denied", "reviewer": decision.reviewer, "note": decision.note}
            else:
                # The model cannot set injected arguments, even if it tries.
                args = {k: v for k, v in call["args"].items() if k not in INJECTED_ARGS}
                if call["name"] == ISSUE_REFUND:
                    args["idempotency_key"] = refund_key(thread_id, call["id"])
                if decision:
                    args["approved_by"] = decision.reviewer
                output = tool.invoke(args, config)
            results.append(
                ToolMessage(content=json.dumps(output), tool_call_id=call["id"], name=call["name"])
            )
        return {"messages": results}

    return run_tools


def _make_escalation(backend: FakePulsoBackend, max_steps: int) -> Callable[..., dict]:
    def escalation(state: SupportState, config: RunnableConfig) -> dict:
        messages = state["messages"]
        patch: list[BaseMessage] = []
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            # Close dangling tool calls so the thread stays valid for the next turn.
            patch = [
                ToolMessage(
                    content=json.dumps({"error": "Not executed: step limit reached."}),
                    tool_call_id=c["id"],
                    name=c["name"],
                )
                for c in last.tool_calls
            ]
        if state.get("route") == "escalation":
            reason = "Request needs a human agent."
        else:
            reason = f"{state.get('route')} agent hit the {max_steps}-step limit."
        last_human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        ticket = backend.open_ticket(thread_id, f"{reason} User said: {last_human.text[:300]}")
        reply = AIMessage(
            content=(
                f"I've passed this to a Pulso support specialist (ticket {ticket}). "
                "They'll reply in this conversation shortly."
            )
        )
        return {"messages": [*patch, reply], "ticket_id": ticket}

    return escalation


def build_support_graph(
    model: BaseChatModel,
    backend: FakePulsoBackend,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    executor: ResilientExecutor | None = None,
) -> CompiledStateGraph:
    settings = settings or Settings()
    policy = RetryPolicy(max_attempts=settings.tool_max_attempts, timeout_s=settings.tool_timeout_s)
    all_tools = build_tools(backend, executor, default_policies(policy))
    graph = StateGraph(SupportState)

    graph.add_node("router", _make_router(model))
    graph.add_node("escalation", _make_escalation(backend, settings.max_agent_steps))
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {route: route for route in prompts.ROUTES},
    )

    for name, tool_names in AGENT_TOOLS.items():
        tools = {t: all_tools[t] for t in tool_names}
        tools_node = f"{name}_tools"
        graph.add_node(name, _make_agent(name, model, list(tools.values())))
        graph.add_node(tools_node, _make_tool_node(tools, settings.refund_approval_threshold_cents))
        graph.add_conditional_edges(
            name,
            _after_agent(tools_node, settings.max_agent_steps),
            {tools_node: tools_node, "escalation": "escalation", END: END},
        )
        graph.add_edge(tools_node, name)

    graph.add_edge("escalation", END)
    return graph.compile(checkpointer=checkpointer)


def _after_agent(tools_node: str, max_steps: int) -> Callable[[SupportState], str]:
    def decide(state: SupportState) -> str:
        last = state["messages"][-1]
        if not (isinstance(last, AIMessage) and last.tool_calls):
            return END
        # Cycle guard: an agent that keeps asking for tools is handed to a human.
        if state.get("steps", 0) >= max_steps:
            return "escalation"
        return tools_node

    return decide


def recursion_limit(settings: Settings) -> int:
    """LangGraph's own super-step limit, set as a backstop above the cycle guard."""
    return 2 * settings.max_agent_steps + 5
