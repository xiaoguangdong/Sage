import unittest

import pandas as pd

from sage_app.pipelines.stock_selector_scheduler import _is_first_trading_day_from_dates


class TestStockSelectorScheduler(unittest.TestCase):
    def test_first_trading_day_true(self):
        dates = pd.Series(["20260203", "20260204", "20260205", "20260206"])
        self.assertTrue(_is_first_trading_day_from_dates(dates, "20260203"))

    def test_first_trading_day_false(self):
        dates = pd.Series(["20260203", "20260204", "20260205", "20260206"])
        self.assertFalse(_is_first_trading_day_from_dates(dates, "20260204"))


if __name__ == "__main__":
    unittest.main()
