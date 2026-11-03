from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from pulso_support import fixtures as fx
from pulso_support import prompts
from pulso_support.config import Settings
from pulso_support.graph import compaction_cut
from pulso_support.llm import ScriptedSupportModel

SMALL = Settings(summarize_after_messages=6, keep_last_messages=2)


def _messages(service, thread):
    return service.graph.get_state(service._config(thread)).values["messages"]


def test_short_threads_are_not_compacted(make_service):
    service = make_service(settings=SMALL)
    service.send("c-0", f"Where is order {fx.ORDER_IN_PROGRESS}?")

    assert service.summary("c-0") is None
    assert len(_messages(service, "c-0")) == 5


def test_old_turns_are_folded_into_a_summary(make_service):
    service = make_service(settings=SMALL)
    service.send("c-1", f"Where is order {fx.ORDER_IN_PROGRESS}?")
    service.send("c-1", "and what about my other order?")
    service.send("c-1", "hello?")

    summary = service.summary("c-1")
    assert fx.ORDER_IN_PROGRESS in summary
    assert "and what about my other order?" in summary
    assert len(_messages(service, "c-1")) <= SMALL.summarize_after_messages


def test_agents_still_resolve_an_order_known_only_from_the_summary(make_service, backend):
    agent_inputs: list[list] = []

    class SpyModel(ScriptedSupportModel):
        def _act(self, messages):
            agent_inputs.append(messages)
            return super()._act(messages)

    service = make_service(model=SpyModel(), settings=SMALL)
    service.send("c-2", f"Where is order {fx.ORDER_DELIVERED}?")
    service.send("c-2", "is there any news?")
    agent_inputs.clear()

    result = service.send("c-2", "Then can I get a $10 refund?")

    first_call = agent_inputs[0]
    mentioning = [m for m in first_call if fx.ORDER_DELIVERED in m.text]
    assert [m.text.startswith(prompts.SUMMARY_PREFIX) for m in mentioning] == [True]
    assert result.tool_calls[0]["args"] == {"order_id": fx.ORDER_DELIVERED}
    assert backend.refunded_cents(fx.ORDER_DELIVERED) == 1_000


def test_summaries_accumulate_across_compactions(make_service):
    service = make_service(settings=SMALL)
    service.send("c-3", f"Where is order {fx.ORDER_IN_PROGRESS}?")
    service.send("c-3", "ok thanks")
    service.send("c-3", f"Where is order {fx.ORDER_DELIVERED}?")
    service.send("c-3", "great")
    service.send("c-3", "bye")

    summary = service.summary("c-3")
    assert fx.ORDER_IN_PROGRESS in summary
    assert fx.ORDER_DELIVERED in summary


def test_summarizer_receives_the_previous_summary(make_service):
    prompts_seen: list[str] = []

    class SpyModel(ScriptedSupportModel):
        def _generate(self, messages, *args, **kwargs):
            if messages[0].text == prompts.SUMMARIZER:
                prompts_seen.append(messages[-1].text)
            return super()._generate(messages, *args, **kwargs)

    service = make_service(model=SpyModel(), settings=SMALL)
    turns = [f"Where is order {fx.ORDER_IN_PROGRESS}?", "ok", "thanks", "hm", "so", "bye"]
    for text in turns:
        service.send("c-4", text)

    assert prompts_seen[0].startswith("Previous summary: (none)")
    assert fx.ORDER_IN_PROGRESS in prompts_seen[-1].split("Messages:")[0]


def test_cut_never_separates_tool_calls_from_their_results():
    call = {"name": "get_order", "args": {}, "id": "c1", "type": "tool_call"}
    messages = [
        HumanMessage("q1", id="1"),
        AIMessage("", tool_calls=[call], id="2"),
        ToolMessage("{}", tool_call_id="c1", id="3"),
        AIMessage("a1", id="4"),
        HumanMessage("q2", id="5"),
        AIMessage("", tool_calls=[{**call, "id": "c2"}], id="6"),
        ToolMessage("{}", tool_call_id="c2", id="7"),
    ]
    # keep_last=2 would start at the ToolMessage; the cut moves back to the user turn.
    assert compaction_cut(messages, keep_last=2) == 4
    assert compaction_cut(messages, keep_last=7) == 0
