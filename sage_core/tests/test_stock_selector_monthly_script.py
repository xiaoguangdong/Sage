import unittest
from types import SimpleNamespace

import pandas as pd

from scripts.stock.run_stock_selector_monthly import _extract_feature_importance, _latest_weekly_trade_date


class DummyRuleSelector:
    def __init__(self):
        self.config = SimpleNamespace(model_type="rule")
        self.feature_cols = ["f1", "f2", "f3"]
        self.rule_weights = {"f1": 0.5, "f2": 0.3, "f3": 0.2}
        self.model = None


class TestStockSelectorMonthlyScript(unittest.TestCase):
    def test_latest_weekly_trade_date(self):
        trade_dates = ["20260105", "20260106", "20260109", "20260112", "20260116"]
        latest = _latest_weekly_trade_date(trade_dates)
        self.assertEqual(latest, "20260116")

    def test_extract_feature_importance_rule(self):
        selector = DummyRuleSelector()
        df = _extract_feature_importance(selector)
        self.assertEqual(list(df.columns), ["feature", "importance", "importance_norm"])
        self.assertEqual(df.iloc[0]["feature"], "f1")
        self.assertAlmostEqual(float(df["importance_norm"].sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
