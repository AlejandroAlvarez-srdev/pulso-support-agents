"""Deterministic input and output guardrails.

These run as graph nodes, outside the model, so they hold no matter which
provider is configured or what a prompt says:

- PII redaction: emails and phone numbers are replaced before any text is stored
  in the checkpoint or sent to the LLM, and again on the way out.
- Prompt-injection screen: messages that try to override instructions or set
  internal tool arguments are refused without reaching the model.
"""

from __future__ import annotations

import re

EMAIL_PLACEHOLDER = "[email redacted]"
PHONE_PLACEHOLDER = "[phone redacted]"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Candidate runs of digits and phone punctuation; validated by digit count below.
# The lookarounds keep UUID segments (bounded by '-' or hex letters) out.
_PHONE_RE = re.compile(r"(?<![\w-])\+?\(?\d[\d\s().-]{6,}\d(?![\w-])")
_PHONE_DIGITS = range(10, 16)

_INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(instructions|rules|prompt|guidelines)\b",
    r"\bdisregard\b.{0,40}\b(instructions|rules|prompt|guidelines|policy)\b",
    r"\b(system|developer)\s+prompt\b",
    r"\byou are now\b",
    r"\b(developer|god|admin)\s+mode\b",
    r"\bjailbreak\b",
    r"\b(approved_by|idempotency_key)\b",
    r"\bact as (an? )?(admin|administrator|developer|system)\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)

INJECTION_REFUSAL = (
    "I can't act on instructions that try to change how I work. I can help with your "
    "Pulso orders, refunds and pulser issues."
)
OUT_OF_SCOPE_REFUSAL = (
    "I can only help with Pulso orders, refunds and pulsers, so I can't help with that. "
    "If you have a question about an order, share its ID and I'll take a look."
)


def _redact_phone(match: re.Match[str]) -> str:
    digits = sum(c.isdigit() for c in match.group(0))
    return PHONE_PLACEHOLDER if digits in _PHONE_DIGITS else match.group(0)


def redact_pii(text: str) -> str:
    return _PHONE_RE.sub(_redact_phone, _EMAIL_RE.sub(EMAIL_PLACEHOLDER, text))


def detect_injection(text: str) -> str | None:
    """Return the matched fragment if the text looks like a prompt-injection attempt."""
    match = _INJECTION_RE.search(text)
    return match.group(0) if match else None


# Trailing text that could still grow into a phone number or email: a run of
# phone characters (digits, spaces, punctuation) followed by the last partial word.
_UNSAFE_TAIL = re.compile(r"[\d\s().+-]*\S*$")


class StreamingRedactor:
    """Applies `redact_pii` to a token stream without splitting a match.

    Text is released only up to the start of the unsafe tail, so an email or
    phone number is always redacted as a whole, at the cost of holding back the
    last word (and any digits before it) until more text arrives.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, token: str) -> str:
        self._buffer += token
        cut = _UNSAFE_TAIL.search(self._buffer).start()
        ready, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return redact_pii(ready)

    def flush(self) -> str:
        ready, self._buffer = self._buffer, ""
        return redact_pii(ready)
