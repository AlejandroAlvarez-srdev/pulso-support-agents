"""Property-based tests for the guardrails.

The streaming redactor promises that the text a client receives token by token
is exactly the text batch redaction produces, for any way the model splits its
output. Hand-picked token splits only cover the splits someone thought of, so
Hypothesis generates the text and the split points.

Example counts come from the Hypothesis profile set in conftest.py:
HYPOTHESIS_PROFILE=stress runs 50,000 examples per property.
"""

from __future__ import annotations

import itertools

from hypothesis import given
from hypothesis import strategies as st

from pulso_support.guardrails import StreamingRedactor, redact_pii

# Fragments that interact with the phone and email patterns: digit runs, phone
# punctuation, '@', dots, hyphens (UUID-like), colons (timestamps) and words.
_FRAGMENTS = st.sampled_from(
    [
        " ",
        "\n",
        "+52",
        "55",
        "1234",
        "5678",
        "(55)",
        "-",
        ".",
        ":",
        "@",
        "\uff20",
        "ana",
        "torres",
        "example",
        "com",
        "mx",
        "order",
        "0b5e1c7a",
        "MXN",
        "$349.00",
        "2026-10-30",
        "14:30",
        "tel:",
        "write to ",
        "call ",
        "\uff15\uff15",
    ]
)
_TEXT = st.lists(_FRAGMENTS, max_size=30).map("".join)


def _chunks(text: str, cuts: list[int]) -> list[str]:
    points = sorted({0, len(text), *(c % (len(text) + 1) for c in cuts)})
    return [text[a:b] for a, b in itertools.pairwise(points)]


@given(text=_TEXT, cuts=st.lists(st.integers(min_value=0), max_size=12))
def test_streamed_redaction_equals_batch_redaction_for_any_split(text, cuts):
    redactor = StreamingRedactor()
    streamed = "".join(redactor.feed(chunk) for chunk in _chunks(text, cuts))
    streamed += redactor.flush()
    assert streamed == redact_pii(text)


@given(text=_TEXT)
def test_redaction_is_idempotent(text):
    once = redact_pii(text)
    assert redact_pii(once) == once


@given(before=_TEXT, after=_TEXT, order_id=st.uuids())
def test_order_ids_survive_redaction_whatever_surrounds_them(before, after, order_id):
    # A word boundary on each side, as in real messages ("order <id>?").
    text = f"{before} {order_id} {after}"
    assert str(order_id) in redact_pii(text)
