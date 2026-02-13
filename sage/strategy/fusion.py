from __future__ import annotations

from typing import List

import pandas as pd

from sage.core.types import Signal


class SignalFusion:
    """
    Fuse signals from macro/trend/rank/entry into executable signals.
    """

    def fuse(
        self,
        trend_state: dict,
        ranked: pd.DataFrame,
        entry_filtered: pd.DataFrame,
        max_positions: int = 20,
    ) -> List[Signal]:
        """
        Args:
            trend_state: output from trend model (dict)
            ranked: DataFrame with columns ['ts_code', 'score']
            entry_filtered: DataFrame subset of ranked (passed filters)
        """
        if trend_state.get("state") == "RISK_OFF":
            return []

        if entry_filtered is None or entry_filtered.empty:
            return []

        top = entry_filtered.head(max_positions)
        weight = round(1.0 / len(top), 4) if len(top) > 0 else 0.0

        signals = []
        for _, row in top.iterrows():
            signals.append(
                Signal(
                    ts_code=row["ts_code"],
                    action="BUY",
                    weight=weight,
                    reason=["SignalFusion: passed filters"],
                )
            )
        return signals

