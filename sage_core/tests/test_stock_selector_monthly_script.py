import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.stock.run_stock_selector_monthly import (
    _evaluate_holdout_predictions,
    _extract_feature_importance,
    _latest_weekly_trade_date,
    _load_industry_map,
    _split_train_valid_by_date,
)


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

    def test_split_train_valid_by_date(self):
        df = pd.DataFrame(
            {
                "trade_date": ["20260101", "20260102", "20260103", "20260104", "20260105"],
                "ts_code": ["000001.SZ"] * 5,
                "close": [10, 11, 12, 13, 14],
            }
        )
        train_df, valid_df = _split_train_valid_by_date(df, valid_days=2)
        self.assertEqual(sorted(train_df["trade_date"].unique().tolist()), ["20260101", "20260102", "20260103"])
        self.assertEqual(sorted(valid_df["trade_date"].unique().tolist()), ["20260104", "20260105"])

    def test_evaluate_holdout_predictions(self):
        rows = []
        dates = pd.date_range("2026-01-01", periods=30, freq="B")
        for idx, date in enumerate(dates):
            for rank, code in enumerate(["000001.SZ", "000002.SZ", "000003.SZ"]):
                close = 10 + idx * 0.1 + rank * 0.05
                rows.append(
                    {
                        "trade_date": date.strftime("%Y%m%d"),
                        "ts_code": code,
                        "close": close,
                    }
                )
        panel = pd.DataFrame(rows)
        pred = panel[["trade_date", "ts_code"]].copy()
        pred["score"] = np.where(pred["ts_code"] == "000001.SZ", 1.0, 0.3)

        metrics = _evaluate_holdout_predictions(panel=panel, pred=pred, top_n=1, label_horizon=5)
        self.assertIn("top_n_excess_return", metrics)
        self.assertIn("rank_ic_mean", metrics)
        self.assertGreaterEqual(metrics["days"], 1.0)

    def test_load_industry_map_supports_ts_code_schema(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            northbound = root / "northbound"
            northbound.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
                    "in_date": ["20200101", "20240101", "20200101"],
                    "out_date": ["", "", ""],
                    "industry_name": ["银行", "银行", "电子"],
                }
            )
            df.to_parquet(northbound / "sw_constituents.parquet", index=False)
            mapping = _load_industry_map(root, "20260213")
            self.assertEqual(set(mapping["ts_code"]), {"000001.SZ", "000002.SZ"})
            self.assertEqual(
                mapping.set_index("ts_code").loc["000002.SZ", "industry_l1"],
                "电子",
            )


if __name__ == "__main__":
    unittest.main()
