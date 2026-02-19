import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector


class TestStockSelector(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        stocks = ["000001.SZ", "000002.SZ", "000003.SZ"]
        industries = {
            "000001.SZ": "801010",
            "000002.SZ": "801010",
            "000003.SZ": "801020",
        }

        rows = []
        for code in stocks:
            price = 10 + np.cumsum(np.random.randn(len(dates)) * 0.2)
            turnover = np.random.uniform(0.01, 0.1, len(dates))
            for i, date in enumerate(dates):
                rows.append(
                    {
                        "trade_date": date.strftime("%Y%m%d"),
                        "ts_code": code,
                        "close": price[i],
                        "turnover": turnover[i],
                        "amount": float(price[i] * 1e6),
                        "pe_ttm": 15.0 + i * 0.02,
                        "pb": 1.2 + i * 0.005,
                        "roe_dt": 8.0 + (i % 20) * 0.1,
                        "grossprofit_margin": 25.0 + (i % 15) * 0.2,
                        "netprofit_yoy": 10.0 + (i % 12),
                        "debt_to_assets": 45.0 + (i % 8),
                        "ocfps": 1.0 + (i % 10) * 0.05,
                        "northbound_hold_ratio": 0.02 + (i % 5) * 0.001,
                        "northbound_net_flow": 1_000_000 + i * 1000,
                        "industry_l1": industries[code],
                    }
                )

        self.df = pd.DataFrame(rows)

    def test_build_labels(self):
        config = SelectionConfig(
            model_type="rule",
            label_horizons=(20,),
            label_weights=(1.0,),
            industry_col="industry_l1",
        )
        selector = StockSelector(config)
        labels = selector.build_labels(self.df)
        valid = labels.dropna()
        self.assertTrue(len(valid) > 0)
        self.assertTrue(((valid >= 0) & (valid <= 1)).all())

    def test_fit_predict_rule(self):
        config = SelectionConfig(
            model_type="rule",
            label_horizons=(5, 10),
            label_weights=(0.7, 0.3),
            industry_col="industry_l1",
        )
        selector = StockSelector(config)
        selector.fit(self.df)
        result = selector.predict(self.df.tail(90))
        self.assertIn("score", result.columns)
        self.assertIn("rank", result.columns)
        self.assertIn("confidence", result.columns)
        self.assertTrue(((result["confidence"] >= 0) & (result["confidence"] <= 1)).all())

    def test_select_top(self):
        selector = StockSelector(SelectionConfig(model_type="rule", label_horizons=(20,), label_weights=(1.0,)))
        selector.fit(self.df)
        last_date = self.df["trade_date"].max()
        picked = selector.select_top(self.df, top_n=2, trade_date=last_date)
        self.assertLessEqual(len(picked), 2)

    def test_feature_inference_excludes_leakage_columns(self):
        selector = StockSelector(SelectionConfig(model_type="rule", label_horizons=(20,), label_weights=(1.0,)))
        feature_df = selector.prepare_features(self.df)
        feature_df["future_return_20d"] = 0.1
        feature_df["target_rank"] = 0.5
        feature_cols = selector._infer_feature_cols(feature_df)
        self.assertTrue(all("future" not in c.lower() for c in feature_cols))
        self.assertTrue(all("target" not in c.lower() for c in feature_cols))
        self.assertLessEqual(len(feature_cols), selector.config.max_feature_count)


if __name__ == "__main__":
    unittest.main()
