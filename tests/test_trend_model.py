import pandas as pd

from sage.models.trend import MovingAverageTrendRule


def test_trend_rule_basic():
    # Create a simple increasing series to force RISK_ON.
    df = pd.DataFrame({"close": list(range(1, 101))})
    model = MovingAverageTrendRule(short_window=5, long_window=10)
    output = model.predict(df)
    assert output.data["state"] in {"RISK_ON", "NEUTRAL"}

