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
        )
