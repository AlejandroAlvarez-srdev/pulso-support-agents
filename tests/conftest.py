from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from hypothesis import settings

from pulso_support.backend import FakePulsoBackend

FIXED_NOW = datetime(2026, 10, 30, 18, 0, tzinfo=UTC)

settings.register_profile("default", max_examples=1000, deadline=None)
settings.register_profile("stress", max_examples=50_000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture
def backend() -> FakePulsoBackend:
    return FakePulsoBackend.with_fixtures(clock=lambda: FIXED_NOW)


@pytest.fixture
def make_service(backend: FakePulsoBackend):
    """Build a SupportService over the shared fake backend with an in-memory checkpointer."""
    from langgraph.checkpoint.memory import InMemorySaver

    from pulso_support.config import Settings
    from pulso_support.graph import build_support_graph
    from pulso_support.llm import ScriptedSupportModel
    from pulso_support.service import SupportService

    def factory(model=None, settings: Settings | None = None, checkpointer=None, executor=None):
        settings = settings or Settings()
        graph = build_support_graph(
            model or ScriptedSupportModel(),
            backend,
            settings,
            checkpointer=checkpointer or InMemorySaver(),
            executor=executor,
        )
        return SupportService(graph, settings)

    return factory
