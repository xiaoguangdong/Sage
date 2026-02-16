import unittest

from scripts.stock.run_stock_model_benchmark import _rank_results, _safe_float


class TestStockModelBenchmarkScript(unittest.TestCase):
    def test_safe_float_handles_invalid_values(self):
        self.assertEqual(_safe_float("1.23"), 1.23)
        self.assertEqual(_safe_float("nan"), float("-inf"))
        self.assertEqual(_safe_float(None), float("-inf"))

    def test_rank_results_orders_by_excess_then_ic(self):
        rows = [
            {"model_type": "rule", "top_n_excess_return": 0.01, "rank_ic_mean": 0.03, "top_n_win_rate": 0.55},
            {"model_type": "lgbm", "top_n_excess_return": 0.02, "rank_ic_mean": 0.01, "top_n_win_rate": 0.50},
            {"model_type": "xgb", "top_n_excess_return": 0.02, "rank_ic_mean": 0.02, "top_n_win_rate": 0.48},
        ]
        ranked = _rank_results(rows)
        self.assertEqual(ranked[0]["model_type"], "xgb")
        self.assertEqual(ranked[0]["benchmark_rank"], 1)
        self.assertEqual(ranked[1]["model_type"], "lgbm")


if __name__ == "__main__":
    unittest.main()
