from __future__ import annotations

import pytest

from sage_core.execution.order_lifecycle import OrderLifecycle, OrderStatus


def test_order_lifecycle_happy_path():
    lifecycle = OrderLifecycle(order_id="ORD-1")
    lifecycle.transit(OrderStatus.ACK, reason="broker accepted")
    lifecycle.transit(OrderStatus.PARTIAL_FILLED, reason="partial")
    lifecycle.transit(OrderStatus.FILLED, reason="done")

    snapshot = lifecycle.snapshot()
    assert snapshot["current_status"] == "FILLED"
    assert len(snapshot["events"]) == 3
    assert snapshot["events"][0]["from_status"] == "NEW"
    assert snapshot["events"][-1]["to_status"] == "FILLED"


def test_order_lifecycle_reject_invalid_transition():
    lifecycle = OrderLifecycle(order_id="ORD-2")
    with pytest.raises(ValueError):
        lifecycle.transit(OrderStatus.FILLED, reason="invalid direct fill")
