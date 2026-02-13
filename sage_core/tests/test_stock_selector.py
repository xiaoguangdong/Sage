import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sage_core.models.stock_selector import StockSelector, SelectionConfig


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
                rows.append({
                    "trade_date": date.strftime("%Y%m%d"),
                    "ts_code": code,
                    "close": price[i],
                    "turnover": turnover[i],
                    "industry_l1": industries[code],
                })

        self.df = pd.DataFrame(rows)

    def test_build_labels(self):
        selector = StockSelector(SelectionConfig(model_type="rule"))
        labels = selector.build_labels(self.df)
        valid = labels.dropna()
        self.assertTrue(len(valid) > 0)
        self.assertTrue(((valid >= 0) & (valid <= 1)).all())

    def test_fit_predict_rule(self):
        config = SelectionConfig(
            model_type="rule",
            label_horizons=(5, 10),
            label_weights=(0.7, 0.3),
        )
        selector = StockSelector(config)
        selector.fit(self.df)
        result = selector.predict(self.df.tail(90))
        self.assertIn("score", result.columns)
        self.assertIn("rank", result.columns)

    def test_select_top(self):
        selector = StockSelector(SelectionConfig(model_type="rule"))
        selector.fit(self.df)
        last_date = self.df["trade_date"].max()
        picked = selector.select_top(self.df, top_n=2, trade_date=last_date)
        self.assertLessEqual(len(picked), 2)


if __name__ == "__main__":
    unittest.main()
