from __future__ import annotations

from typing import List

from sage.core.types import Signal


class RiskController:
    """
    Apply position/industry/portfolio constraints.
    """

    def __init__(
        self,
        max_position_weight: float = 0.1,
        max_industry_weight: float = 0.3,
        max_total_weight: float = 1.0,
    ) -> None:
        self.max_position_weight = max_position_weight
        self.max_industry_weight = max_industry_weight
        self.max_total_weight = max_total_weight

    def apply(self, signals: List[Signal]) -> List[Signal]:
        # Placeholder: implement industry-aware capping later.
        capped = []
        for s in signals:
            weight = min(s.weight, self.max_position_weight)
            capped.append(Signal(s.ts_code, s.action, weight, s.reason))
        return capped

