"""Pulso domain model, shared in shape with the rest of the Pulso repos.

Python attributes are snake_case; serialised JSON uses the camelCase field
names of the shared contract (``merchantId``, ``amountCents``...).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class _Model(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)

    def to_json(self) -> dict:
        return self.model_dump(mode="json", by_alias=True)


class OrderStatus(StrEnum):
    CREATED = "CREATED"
    PAYMENT_PENDING = "PAYMENT_PENDING"
    PAID = "PAID"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class Vehicle(StrEnum):
    BIKE = "BIKE"
    MOTO = "MOTO"
    CAR = "CAR"


class Location(_Model):
    lat: float
    lng: float
    address: str | None = None


class Item(_Model):
    sku: str
    name: str
    qty: int = Field(gt=0)


class Order(_Model):
    id: str
    merchant_id: str
    status: OrderStatus
    pickup: Location
    dropoff: Location
    items: list[Item]
    amount_cents: int = Field(ge=0)
    currency: str = "MXN"
    created_at: datetime


class Pulser(_Model):
    id: str
    name: str
    location: Location
    rating: float = Field(ge=0, le=5)
    active_tasks: int = Field(ge=0)
    max_tasks: int = Field(gt=0)
    vehicle: Vehicle
    online: bool

    @property
    def has_capacity(self) -> bool:
        return self.online and self.active_tasks < self.max_tasks


class Assignment(_Model):
    order_id: str
    pulser_id: str
    score: float
    eta_minutes: int
    assigned_at: datetime


class Refund(_Model):
    id: str
    order_id: str
    amount_cents: int
    currency: str = "MXN"
    reason: str
    approved_by: str | None = None
    idempotency_key: str | None = None
    issued_at: datetime


class Event(_Model):
    type: str
    payload: dict
    occurred_at: datetime


# Statuses in which money has been captured and can be returned.
REFUNDABLE_STATUSES = frozenset(
    {
        OrderStatus.PAID,
        OrderStatus.ASSIGNED,
        OrderStatus.IN_PROGRESS,
        OrderStatus.DELIVERED,
        OrderStatus.CANCELLED,
        OrderStatus.FAILED,
    }
)

# Statuses in which a different pulser can still take over the order.
REASSIGNABLE_STATUSES = frozenset({OrderStatus.PAID, OrderStatus.ASSIGNED})
