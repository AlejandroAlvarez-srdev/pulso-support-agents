"""Refund ledger with idempotency keys, in memory or in SQLite.

The SQLite ledger is what makes refunds safe across processes: if a worker
crashes after the refund is written but before the graph checkpoints, the
resumed run replays the same tool call with the same key and gets the original
refund back instead of paying twice.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Protocol

from pulso_support.domain import Refund


class RefundLedger(Protocol):
    def transaction(self) -> AbstractContextManager[None]: ...
    def by_key(self, key: str) -> Refund | None: ...
    def for_order(self, order_id: str) -> list[Refund]: ...
    def next_id(self) -> str: ...
    def add(self, refund: Refund) -> None: ...


class InMemoryRefundLedger:
    def __init__(self) -> None:
        self._refunds: list[Refund] = []
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self._lock:
            yield

    def by_key(self, key: str) -> Refund | None:
        return next((r for r in self._refunds if r.idempotency_key == key), None)

    def for_order(self, order_id: str) -> list[Refund]:
        return [r for r in self._refunds if r.order_id == order_id]

    def next_id(self) -> str:
        return f"rf_{len(self._refunds) + 1:04d}"

    def add(self, refund: Refund) -> None:
        self._refunds.append(refund)


class SqliteRefundLedger:
    def __init__(self, path: str) -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._lock = threading.RLock()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS refunds (
                   id TEXT PRIMARY KEY,
                   idempotency_key TEXT UNIQUE,
                   order_id TEXT NOT NULL,
                   payload TEXT NOT NULL
               )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS refunds_order ON refunds(order_id)")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        # BEGIN IMMEDIATE takes the write lock up front, so check-then-insert is
        # atomic across processes sharing the file, not only across threads.
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def by_key(self, key: str) -> Refund | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM refunds WHERE idempotency_key = ?", (key,)
            ).fetchone()
        return Refund.model_validate_json(row[0]) if row else None

    def for_order(self, order_id: str) -> list[Refund]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM refunds WHERE order_id = ? ORDER BY rowid", (order_id,)
            ).fetchall()
        return [Refund.model_validate_json(r[0]) for r in rows]

    def next_id(self) -> str:
        with self._lock:
            (count,) = self._conn.execute("SELECT COUNT(*) FROM refunds").fetchone()
        return f"rf_{count + 1:04d}"

    def add(self, refund: Refund) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO refunds (id, idempotency_key, order_id, payload) VALUES (?, ?, ?, ?)",
                (
                    refund.id,
                    refund.idempotency_key,
                    refund.order_id,
                    refund.model_dump_json(by_alias=True),
                ),
            )
