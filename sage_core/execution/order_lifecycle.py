from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class OrderStatus(str, Enum):
    NEW = "NEW"
    ACK = "ACK"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


ALLOWED_TRANSITIONS: Dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.NEW: {OrderStatus.ACK, OrderStatus.REJECTED},
    OrderStatus.ACK: {OrderStatus.PARTIAL_FILLED, OrderStatus.FILLED, OrderStatus.CANCEL_PENDING, OrderStatus.REJECTED},
    OrderStatus.PARTIAL_FILLED: {OrderStatus.FILLED, OrderStatus.CANCEL_PENDING},
    OrderStatus.CANCEL_PENDING: {OrderStatus.CANCELLED, OrderStatus.PARTIAL_FILLED, OrderStatus.FILLED},
    OrderStatus.FILLED: set(),
    OrderStatus.CANCELLED: set(),
    OrderStatus.REJECTED: set(),
}


@dataclass
class OrderStateEvent:
    order_id: str
    from_status: OrderStatus
    to_status: OrderStatus
    event_time: str
    reason: str = ""

    def to_dict(self) -> Dict[str, str]:
        data = asdict(self)
        data["from_status"] = self.from_status.value
        data["to_status"] = self.to_status.value
        return data


class OrderLifecycle:
    def __init__(self, order_id: str, initial_status: OrderStatus = OrderStatus.NEW) -> None:
        self.order_id = str(order_id)
        self.current_status = initial_status
        self.events: List[OrderStateEvent] = []

    def can_transit(self, to_status: OrderStatus) -> bool:
        return to_status in ALLOWED_TRANSITIONS[self.current_status]

    def transit(self, to_status: OrderStatus, reason: str = "", event_time: Optional[str] = None) -> OrderStateEvent:
        if not self.can_transit(to_status):
            raise ValueError(
                f"非法状态流转: {self.current_status.value} -> {to_status.value} (order_id={self.order_id})"
            )
        timestamp = event_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event = OrderStateEvent(
            order_id=self.order_id,
            from_status=self.current_status,
            to_status=to_status,
            event_time=timestamp,
            reason=reason,
        )
        self.events.append(event)
        self.current_status = to_status
        return event

    def snapshot(self) -> Dict[str, object]:
        return {
            "order_id": self.order_id,
            "current_status": self.current_status.value,
            "events": [event.to_dict() for event in self.events],
        }
