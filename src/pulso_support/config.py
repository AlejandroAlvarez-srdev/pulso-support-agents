"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Refunds strictly above this amount pause the graph for a human reviewer.
    refund_approval_threshold_cents: int = 50_000
    # Agent invocations allowed per user turn before the cycle guard escalates.
    max_agent_steps: int = 6
    # "scripted" (offline, deterministic), "anthropic" or "openai".
    llm_provider: str = "scripted"
    llm_model: str | None = None
    # SQLite file for the checkpointer; None keeps threads in memory.
    checkpoint_db: str | None = None
    # Memory compaction: once a thread holds more than this many messages, older
    # turns are folded into a running summary and only the newest ones are kept.
    summarize_after_messages: int = 40
    keep_last_messages: int = 10
    # Tracing: append each run's spans to a JSONL file and/or export to OpenTelemetry.
    trace_file: str | None = None
    otel_enabled: bool = False
    # SQLite file for the fake backend's refund ledger; None keeps it in memory.
    backend_db: str | None = None
    # Per-attempt timeout and attempt budget for backend calls made by tools.
    tool_timeout_s: float = 2.0
    tool_max_attempts: int = 3

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            refund_approval_threshold_cents=int(
                os.getenv("PULSO_REFUND_THRESHOLD_CENTS", cls.refund_approval_threshold_cents)
            ),
            max_agent_steps=int(os.getenv("PULSO_MAX_AGENT_STEPS", cls.max_agent_steps)),
            llm_provider=os.getenv("PULSO_LLM_PROVIDER", cls.llm_provider).lower(),
            llm_model=os.getenv("PULSO_LLM_MODEL") or None,
            checkpoint_db=os.getenv("PULSO_CHECKPOINT_DB") or None,
            summarize_after_messages=int(
                os.getenv("PULSO_SUMMARIZE_AFTER_MESSAGES", cls.summarize_after_messages)
            ),
            keep_last_messages=int(os.getenv("PULSO_KEEP_LAST_MESSAGES", cls.keep_last_messages)),
            trace_file=os.getenv("PULSO_TRACE_FILE") or None,
            otel_enabled=os.getenv("PULSO_OTEL_ENABLED", "").lower() in {"1", "true", "yes"},
            backend_db=os.getenv("PULSO_BACKEND_DB") or None,
            tool_timeout_s=float(os.getenv("PULSO_TOOL_TIMEOUT_S", cls.tool_timeout_s)),
            tool_max_attempts=int(os.getenv("PULSO_TOOL_MAX_ATTEMPTS", cls.tool_max_attempts)),
        )
