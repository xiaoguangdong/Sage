from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import pandas as pd

from sage.core.types import ModelOutput, TrendState
from sage.models.base import RuleBasedModel


class MovingAverageTrendRule(RuleBasedModel):
    """
    Simple moving-average based trend model.

    Expected input DataFrame columns:
      - close: index/benchmark close price
      - volatility (optional): precomputed volatility series
    """

    name = "trend_ma_rule"

    def __init__(self, short_window: int = 20, long_window: int = 60) -> None:
        self.short_window = short_window
        self.long_window = long_window

    def _compute_state(self, df: pd.DataFrame) -> TrendState:
        if "close" not in df.columns:
            raise ValueError("Input DataFrame must contain 'close' column")

        close = df["close"]
        ma_short = close.rolling(self.short_window).mean()
        ma_long = close.rolling(self.long_window).mean()

        latest_short = ma_short.iloc[-1]
        latest_long = ma_long.iloc[-1]

        reason = []
        state = "NEUTRAL"
        confidence = 0.5
        position_suggestion = 0.5

        if latest_short > latest_long:
            state = "RISK_ON"
            confidence = 0.7
            position_suggestion = 0.8
            reason.append("MA trend up")
        else:
            state = "RISK_OFF"
            confidence = 0.7
            position_suggestion = 0.2
            reason.append("MA trend down")

        # Optional volatility filter
        if "volatility" in df.columns:
            vol = df["volatility"].iloc[-1]
            # Simple threshold rule (to be tuned with backtests)
            if vol > df["volatility"].median():
                reason.append("Volatility elevated")
                if state == "RISK_ON":
                    state = "NEUTRAL"
                    confidence = min(confidence, 0.6)
                    position_suggestion = min(position_suggestion, 0.5)

        return TrendState(
            state=state,
            confidence=float(confidence),
            position_suggestion=float(position_suggestion),
            reason=reason,
        )

    def predict(self, dataset: pd.DataFrame) -> ModelOutput:
        state = self._compute_state(dataset)
        return ModelOutput(name=self.name, data=asdict(state))

