from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pulso_support.backend import FakePulsoBackend

FIXED_NOW = datetime(2026, 10, 30, 18, 0, tzinfo=UTC)


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

    def factory(model=None, settings: Settings | None = None, checkpointer=None):
        settings = settings or Settings()
        graph = build_support_graph(
            model or ScriptedSupportModel(),
            backend,
            settings,
            checkpointer=checkpointer or InMemorySaver(),
        )
        return SupportService(graph, settings)

    return factory
