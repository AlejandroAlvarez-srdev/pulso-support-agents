from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pulso_support.backend import FakePulsoBackend

FIXED_NOW = datetime(2026, 10, 30, 18, 0, tzinfo=UTC)


@pytest.fixture
def backend() -> FakePulsoBackend:
    return FakePulsoBackend.with_fixtures(clock=lambda: FIXED_NOW)
