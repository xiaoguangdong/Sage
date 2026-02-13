from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


@dataclass
class ModelOutput:
    """
    Standard output for all models.

    Attributes:
        name: Model name or id.
        data: Main output payload (e.g., DataFrame, dict, list).
        meta: Auxiliary metadata for debugging and traceability.
    """
    name: str
    data: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal:
    """
    Trading signal produced by a strategy or model.

    Attributes:
        ts_code: A-share stock code.
        action: BUY/SELL/HOLD.
        weight: Suggested portfolio weight.
        reason: Short human-readable reason list.
    """
    ts_code: str
    action: str
    weight: float
    reason: Iterable[str] = field(default_factory=list)


@dataclass
class TrendState:
    """
    Market regime output.
    """
    state: str  # RISK_ON / RISK_OFF / NEUTRAL
    confidence: float
    position_suggestion: float
    reason: Iterable[str] = field(default_factory=list)


class BaseModel:
    """
    Base interface for all models (macro/trend/rank/entry).
    """
    name: str = "base-model"

    def fit(self, dataset: Any) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    def predict(self, dataset: Any) -> ModelOutput:  # pragma: no cover - interface only
        raise NotImplementedError

