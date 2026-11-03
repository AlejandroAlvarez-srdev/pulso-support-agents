"""Deterministic seed data for the in-process fake Pulso backend (Mexico City)."""

from __future__ import annotations

from datetime import UTC, datetime

from pulso_support.domain import Assignment, Item, Location, Order, OrderStatus, Pulser, Vehicle

MERCHANT_TAQUERIA = "5a1f0c3e-7b2d-4e8a-9f10-1c2d3e4f5a01"
MERCHANT_FARMACIA = "5a1f0c3e-7b2d-4e8a-9f10-1c2d3e4f5a02"

ORDER_DELIVERED = "0b5e1c7a-2f43-4c8e-9a1d-6f2e3b4c5d01"
ORDER_IN_PROGRESS = "0b5e1c7a-2f43-4c8e-9a1d-6f2e3b4c5d02"
ORDER_ASSIGNED_STUCK = "0b5e1c7a-2f43-4c8e-9a1d-6f2e3b4c5d03"
ORDER_DELIVERED_LARGE = "0b5e1c7a-2f43-4c8e-9a1d-6f2e3b4c5d04"
ORDER_PAYMENT_PENDING = "0b5e1c7a-2f43-4c8e-9a1d-6f2e3b4c5d05"
ORDER_CANCELLED = "0b5e1c7a-2f43-4c8e-9a1d-6f2e3b4c5d06"

PULSER_ANA = "9c3d2b1a-0e4f-4a5b-8c6d-7e8f9a0b1c01"
PULSER_LUIS = "9c3d2b1a-0e4f-4a5b-8c6d-7e8f9a0b1c02"
PULSER_SOFIA = "9c3d2b1a-0e4f-4a5b-8c6d-7e8f9a0b1c03"
PULSER_DIEGO = "9c3d2b1a-0e4f-4a5b-8c6d-7e8f9a0b1c04"
PULSER_MARIANA = "9c3d2b1a-0e4f-4a5b-8c6d-7e8f9a0b1c05"

_T0 = datetime(2026, 10, 30, 17, 0, tzinfo=UTC)

_ROMA = Location(lat=19.4194, lng=-99.1617, address="Av. Álvaro Obregón 120, Roma Nte.")
_CONDESA = Location(lat=19.4111, lng=-99.1733, address="Av. Michoacán 78, Condesa")
_POLANCO = Location(lat=19.4326, lng=-99.1994, address="Av. Presidente Masaryk 340, Polanco")
_DEL_VALLE = Location(lat=19.3846, lng=-99.1640, address="Av. Coyoacán 1435, Del Valle")
_CENTRO = Location(lat=19.4326, lng=-99.1332, address="Av. 5 de Mayo 32, Centro")


def _order(
    order_id: str,
    merchant: str,
    status: OrderStatus,
    amount: int,
    items: list[Item],
    pickup: Location,
    dropoff: Location,
    minutes: int,
) -> Order:
    return Order(
        id=order_id,
        merchant_id=merchant,
        status=status,
        pickup=pickup,
        dropoff=dropoff,
        items=items,
        amount_cents=amount,
        created_at=_T0.replace(minute=minutes),
    )


def seed_orders() -> list[Order]:
    tacos = [Item(sku="TAC-PASTOR", name="Tacos al pastor x10", qty=2)]
    meds = [Item(sku="MED-PARA-500", name="Paracetamol 500mg", qty=1)]
    return [
        _order(
            ORDER_DELIVERED,
            MERCHANT_TAQUERIA,
            OrderStatus.DELIVERED,
            34_900,
            tacos,
            _ROMA,
            _CONDESA,
            1,
        ),
        _order(
            ORDER_IN_PROGRESS,
            MERCHANT_TAQUERIA,
            OrderStatus.IN_PROGRESS,
            27_500,
            tacos,
            _ROMA,
            _DEL_VALLE,
            5,
        ),
        _order(
            ORDER_ASSIGNED_STUCK,
            MERCHANT_FARMACIA,
            OrderStatus.ASSIGNED,
            15_800,
            meds,
            _POLANCO,
            _CENTRO,
            10,
        ),
        _order(
            ORDER_DELIVERED_LARGE,
            MERCHANT_FARMACIA,
            OrderStatus.DELIVERED,
            189_000,
            [Item(sku="MED-KIT-PRO", name="Botiquín profesional", qty=3)],
            _POLANCO,
            _ROMA,
            15,
        ),
        _order(
            ORDER_PAYMENT_PENDING,
            MERCHANT_TAQUERIA,
            OrderStatus.PAYMENT_PENDING,
            19_900,
            tacos,
            _ROMA,
            _CENTRO,
            20,
        ),
        _order(
            ORDER_CANCELLED,
            MERCHANT_FARMACIA,
            OrderStatus.CANCELLED,
            52_000,
            meds,
            _POLANCO,
            _DEL_VALLE,
            25,
        ),
    ]


def seed_pulsers() -> list[Pulser]:
    return [
        Pulser(
            id=PULSER_ANA,
            name="Ana Torres",
            location=_DEL_VALLE,
            rating=4.9,
            active_tasks=1,
            max_tasks=3,
            vehicle=Vehicle.MOTO,
            online=True,
        ),
        Pulser(
            id=PULSER_LUIS,
            name="Luis Ramírez",
            location=_POLANCO,
            rating=4.1,
            active_tasks=1,
            max_tasks=2,
            vehicle=Vehicle.BIKE,
            online=False,
        ),
        Pulser(
            id=PULSER_SOFIA,
            name="Sofía Méndez",
            location=_CONDESA,
            rating=4.7,
            active_tasks=0,
            max_tasks=2,
            vehicle=Vehicle.BIKE,
            online=True,
        ),
        Pulser(
            id=PULSER_DIEGO,
            name="Diego Herrera",
            location=_POLANCO,
            rating=4.4,
            active_tasks=0,
            max_tasks=4,
            vehicle=Vehicle.CAR,
            online=True,
        ),
        Pulser(
            id=PULSER_MARIANA,
            name="Mariana Cruz",
            location=_CENTRO,
            rating=4.8,
            active_tasks=2,
            max_tasks=2,
            vehicle=Vehicle.MOTO,
            online=True,
        ),
    ]


def seed_assignments() -> list[Assignment]:
    return [
        Assignment(
            order_id=ORDER_DELIVERED,
            pulser_id=PULSER_SOFIA,
            score=0.91,
            eta_minutes=12,
            assigned_at=_T0.replace(minute=2),
        ),
        Assignment(
            order_id=ORDER_IN_PROGRESS,
            pulser_id=PULSER_ANA,
            score=0.88,
            eta_minutes=18,
            assigned_at=_T0.replace(minute=6),
        ),
        Assignment(
            order_id=ORDER_ASSIGNED_STUCK,
            pulser_id=PULSER_LUIS,
            score=0.74,
            eta_minutes=22,
            assigned_at=_T0.replace(minute=11),
        ),
        Assignment(
            order_id=ORDER_DELIVERED_LARGE,
            pulser_id=PULSER_DIEGO,
            score=0.83,
            eta_minutes=15,
            assigned_at=_T0.replace(minute=16),
        ),
    ]
