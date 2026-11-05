"""Deterministic input and output guardrails.

These run as graph nodes, outside the model, so they hold no matter which
provider is configured or what a prompt says:

- PII redaction: emails and phone numbers are replaced before any text is stored
  in the checkpoint or sent to the LLM, and again on the way out.
- Prompt-injection screen: messages that try to override instructions or set
  internal tool arguments are refused without reaching the model. Patterns are
  matched in English and Spanish on a normalised copy of the text (NFKC, invisible
  format characters removed, accents stripped), so zero-width spaces, soft hyphens
  and fullwidth letters don't slip past them.

Both are scored against a labelled corpus by ``pulso_support.guardrail_evals``.
"""

from __future__ import annotations

import re
import unicodedata

EMAIL_PLACEHOLDER = "[email redacted]"
PHONE_PLACEHOLDER = "[phone redacted]"

# The fullwidth at sign (U+FF20) renders like '@' and is the cheapest way around
# a plain regex.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+[@\uff20][A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Candidate runs of digits and phone punctuation; validated by digit count below.
# The lookarounds keep UUID segments (bounded by '-' or hex letters) out, and a
# run touching ':' and a digit on either side is part of a timestamp
# ("2026-10-30 14:30"), not a phone. Keeping both sides symmetric makes
# redaction idempotent: redacting one run never exposes its neighbour.
_PHONE_START = r"(?<![\w-])(?<!\d:)\+?\(?\d"
_PHONE_RE = re.compile(_PHONE_START + r"[\d\s().-]{6,}\d(?![\w-]|:\d)")
_PHONE_DIGITS = range(10, 16)

# Override verbs only count when they target the assistant's own instructions
# ("ignore your/all previous rules"), not the customer's ("don't ignore the
# delivery instructions"). Role changes only count when they name a role.
_ROLE = (
    r"\w*(bot|gpt|ai|assistant|admin|administrator|developer|supervisor|manager|lead"
    r"|operator|system)"
)
_INJECTION_PATTERNS = [
    # English
    r"\b(ignore|disregard|forget|override)\b.{0,20}"
    r"\b(your|all|any|previous|prior|earlier|above|system)\b.{0,20}"
    r"\b(instructions|rules|prompt|guidelines|polic(y|ies))\b",
    r"\bnew (instructions|rules)\s*:\s*(from now on|you|always|never|ignore)\b",
    r"\b(system|developer)\s+prompt\b",
    r"\byour\s+(hidden|secret|original|initial)\s+(instructions|prompt|rules)\b",
    r"\b(print|repeat|output)\s+(everything|all|the (whole )?(text|conversation))\b"
    r".{0,20}\babove\b",
    rf"\byou are now (an? |the )?{_ROLE}\b",
    rf"\bpretend (you are|you're|to be) (an? |the )?(\w+ )?{_ROLE}\b",
    r"\b(developer|god|admin)\s+mode\b",
    r"\bjailbreak\b",
    r"\bact as (an? )?(admin|administrator|developer|system)\b",
    # Spanish (matched without accents)
    r"\b(ignora|ignorar|ignoren|olvida|olvidar|olviden|omite|omitir)\b.{0,20}"
    r"(\b(tus|sus|todas|todos)\b.{0,20}\b(instrucciones|reglas|indicaciones|politicas?)\b"
    r"|\b(instrucciones|reglas|indicaciones|politicas?)\s+"
    r"(anteriores|previas|originales|del sistema)\b)",
    r"\bprompt (de|del) sistema\b",
    r"\bahora eres (un |una |el |la )?\w*(bot|ia|asistente|admin|administrador|desarrollador)",
    r"\bmodo (desarrollador|dios|admin|administrador)\b",
    r"\bactua como (un |una )?(admin|administrador|desarrollador|sistema)\b",
    # Internal tool arguments the model never exposes
    r"\b(approved_by|idempotency_key)\b",
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


def normalize_for_screening(text: str) -> str:
    """Fold the text so look-alike spellings hit the same patterns.

    NFKC maps fullwidth and compatibility forms to ASCII, invisible format
    characters (zero-width spaces, soft hyphens, bidi marks) are dropped, and
    accents are stripped so "actúa" and "actua" match alike. Homoglyphs from
    other scripts (a Cyrillic small o) are not folded; that needs a confusables table.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def detect_injection(text: str) -> str | None:
    """Return the matched fragment if the text looks like a prompt-injection attempt."""
    match = _INJECTION_RE.search(normalize_for_screening(text))
    return match.group(0) if match else None


# Trailing text that could still grow into a phone number or email, or change how
# one is matched. Either the last word, or a phone candidate that runs to the end
# of the buffer, whichever starts first. The candidate starts where the phone
# pattern itself could start (its lookbehind included), not at a space: in
# "$349.00 2026-10-30" the batch pattern matches from the 3, and in
# "ana2026-10-30 55" it can't match at all. Cutting anywhere else would redact
# the held-back text without the context that decides whether it is a phone.
_UNSAFE_TAIL = re.compile(_PHONE_START + r"[\d\s().-]*\S*$|(?<!\S)\S*$")


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
