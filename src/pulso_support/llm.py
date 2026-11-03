"""Chat-model factory plus a deterministic scripted model for offline runs.

The graph only depends on LangChain's ``BaseChatModel`` interface (``invoke`` and
``bind_tools``), so any provider works. ``ScriptedSupportModel`` implements the same
interface with keyword rules: it lets the test-suite, the eval harness and the demo
exercise routing, tool calling and human-in-the-loop with no API key and no network.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from pulso_support import prompts
from pulso_support.config import Settings
from pulso_support.tools import GET_ASSIGNMENT, GET_ORDER, ISSUE_REFUND, REASSIGN_ORDER

UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_AMOUNT_RE = re.compile(
    r"(?:\$\s*(?P<a>\d+(?:[.,]\d{1,2})?))|(?:(?P<b>\d+(?:[.,]\d{1,2})?)\s*(?:mxn|pesos)\b)",
    re.I,
)

# Ordered: the first matching route wins, so specific intents beat generic "order" talk.
_ROUTE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("escalation", ("human", "real person", "manager", "lawyer", "legal", "complaint")),
    ("refunds", ("refund", "money back", "reembolso", "charged twice", "overcharged")),
    (
        "pulser_ops",
        (
            "reassign",
            "another pulser",
            "different pulser",
            "pulser is offline",
            "pulser is not",
            "pulser hasn't",
            "no-show",
            "never showed",
        ),
    ),
    ("order_status", ("where is", "status", "track", "eta", "when will", "arrive", "on its way")),
)


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Build the configured chat model. Provider SDKs are imported lazily."""
    provider = settings.llm_provider
    if provider == "scripted":
        return ScriptedSupportModel()
    if not settings.llm_model:
        raise ValueError(f"PULSO_LLM_MODEL must be set when PULSO_LLM_PROVIDER={provider}")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=settings.llm_model, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.llm_model, temperature=0)
    raise ValueError(f"Unknown PULSO_LLM_PROVIDER: {provider!r}")


# Words that make an otherwise unmatched message a Pulso matter for a human.
_DOMAIN_WORDS = ("order", "pulser", "pedido", "delivery", "payment", "charge", "account", "pulso")


def classify_by_keywords(text: str) -> str:
    lowered = text.lower()
    for route, keywords in _ROUTE_KEYWORDS:
        if any(k in lowered for k in keywords):
            return route
    if UUID_RE.search(text) or any(w in lowered for w in _DOMAIN_WORDS):
        return "escalation"
    return "out_of_scope"


class ScriptedSupportModel(BaseChatModel):
    """Rule-based stand-in for an LLM.

    Without tools bound it acts as the router and answers with a route label.
    With tools bound it plans tool calls for the current turn from the tools it
    was given, then writes a reply from the tool results.
    """

    tool_names: tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "pulso-scripted"

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> ScriptedSupportModel:
        return self.model_copy(update={"tool_names": tuple(t.name for t in tools)})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if messages and messages[0].text == prompts.SUMMARIZER:
            message = self._summarize_history(messages[-1].text)
        elif not self.tool_names:
            message = self._route(messages)
        else:
            message = self._act(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # --- memory compaction ------------------------------------------------

    @staticmethod
    def _summarize_history(transcript: str) -> AIMessage:
        # Tool output also holds merchant and pulser UUIDs; only conversational lines
        # (and the previous summary) name the orders the user cares about.
        conversational = [
            line
            for line in transcript.splitlines()
            if line.startswith(("Previous summary:", "user: ", "assistant"))
        ]
        found = UUID_RE.findall("\n".join(conversational))
        order_ids = list(dict.fromkeys(m.lower() for m in found))
        requests = [
            line.removeprefix("user: ")[:120]
            for line in transcript.splitlines()
            if line.startswith("user: ")
        ]
        parts = []
        if order_ids:
            parts.append(f"Orders discussed: {', '.join(order_ids)}.")
        if requests:
            parts.append("Recent user requests: " + "; ".join(requests[-3:]) + ".")
        return AIMessage(" ".join(parts) or "No relevant history.")

    # --- router ----------------------------------------------------------

    @staticmethod
    def _route(messages: list[BaseMessage]) -> AIMessage:
        human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        return AIMessage(content=classify_by_keywords(human.text))

    # --- agents ----------------------------------------------------------

    def _act(self, messages: list[BaseMessage]) -> AIMessage:
        turn_start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        turn = messages[turn_start:]
        results = {m.name: _parse(m) for m in turn if isinstance(m, ToolMessage)}
        order_id = _find_order_id(messages)
        tools = set(self.tool_names)

        if order_id is None:
            return AIMessage(content="Could you share the order ID (a UUID) so I can look it up?")

        if ISSUE_REFUND in tools:
            if GET_ORDER not in results:
                return _call(messages, (GET_ORDER, {"order_id": order_id}))
            order = results[GET_ORDER]
            if ISSUE_REFUND not in results and "error" not in order:
                remaining = order["amountCents"] - order["refundedCents"]
                amount = _requested_amount_cents(turn[0].text) or remaining
                reason = turn[0].text.strip()[:200]
                return _call(
                    messages,
                    (
                        ISSUE_REFUND,
                        {"order_id": order_id, "amount_cents": amount, "reason": reason},
                    ),
                )
            return AIMessage(content=_summarize(results))

        if {GET_ORDER, GET_ASSIGNMENT} <= tools:
            if not results:
                args = {"order_id": order_id}
                return _call(messages, (GET_ORDER, args), (GET_ASSIGNMENT, args))
            return AIMessage(content=_summarize(results))

        if REASSIGN_ORDER in tools:
            if not results:
                reason = turn[0].text.strip()[:200]
                return _call(messages, (REASSIGN_ORDER, {"order_id": order_id, "reason": reason}))
            return AIMessage(content=_summarize(results))

        return AIMessage(content="I can't help with that here; let me get a specialist.")


def _call(history: list[BaseMessage], *calls: tuple[str, dict]) -> AIMessage:
    # Ids only need to be unique within a thread; history length keeps them deterministic.
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{len(history)}_{i}", "type": "tool_call"}
            for i, (name, args) in enumerate(calls)
        ],
    )


def _parse(message: ToolMessage) -> dict:
    try:
        parsed = json.loads(message.text)
    except json.JSONDecodeError:
        return {"error": message.text}
    return parsed if isinstance(parsed, dict) else {"result": parsed}


def _find_order_id(messages: list[BaseMessage]) -> str | None:
    """Most recent order id mentioned in the thread, including its compacted summary."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage | AIMessage | SystemMessage) and (
            match := UUID_RE.search(message.text)
        ):
            return match.group(0).lower()
    return None


def _requested_amount_cents(text: str) -> int | None:
    match = _AMOUNT_RE.search(text)
    if not match:
        return None
    raw = (match.group("a") or match.group("b")).replace(",", ".")
    return round(float(raw) * 100)


def _mxn(cents: int) -> str:
    return f"${cents / 100:,.2f} MXN"


def _summarize(results: dict[str, dict]) -> str:
    if any(r.get("retryable") for r in results.values()):
        return (
            "Pulso's systems are temporarily unavailable, so I couldn't complete that. "
            "Nothing was changed; please try again in a few minutes."
        )
    parts: list[str] = []
    order = results.get(GET_ORDER)
    if order is not None:
        if "error" in order:
            return f"I couldn't find that order: {order['error']}"
        parts.append(f"Order {order['id']} is {order['status']} ({_mxn(order['amountCents'])}).")
    if (assignment := results.get(GET_ASSIGNMENT)) is not None:
        if "pulser" in assignment:
            p = assignment["pulser"]
            parts.append(
                f"{p['name']} ({p['vehicle']}) is assigned, ETA {assignment['etaMinutes']} min."
            )
        else:
            parts.append("No pulser has been assigned yet.")
    if (refund := results.get(ISSUE_REFUND)) is not None:
        if refund.get("status") == "denied":
            parts.append("The refund was reviewed and not approved; a specialist will follow up.")
        elif "error" in refund:
            parts.append(f"I couldn't issue the refund: {refund['error']}")
        else:
            parts.append(f"Refund {refund['id']} of {_mxn(refund['amountCents'])} has been issued.")
    if (reassigned := results.get(REASSIGN_ORDER)) is not None:
        if "error" in reassigned:
            parts.append(f"I couldn't reassign the order: {reassigned['error']}")
        else:
            p = reassigned["pulser"]
            parts.append(
                f"Order {reassigned['orderId']} is now with {p['name']} ({p['vehicle']}), "
                f"ETA {reassigned['etaMinutes']} min."
            )
    return " ".join(parts)
