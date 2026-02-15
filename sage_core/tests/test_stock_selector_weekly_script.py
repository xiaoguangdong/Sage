import tempfile
import unittest
from pathlib import Path

from scripts.stock.run_stock_selector_weekly_signal import _find_latest_model_metadata


class TestStockSelectorWeeklyScript(unittest.TestCase):
    def test_find_latest_model_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "stock_selector_rule_20260131.meta.json").write_text("{}", encoding="utf-8")
            (model_dir / "stock_selector_rule_20260228.meta.json").write_text("{}", encoding="utf-8")
            latest = _find_latest_model_metadata(model_dir)
            self.assertTrue(str(latest).endswith("stock_selector_rule_20260228.meta.json"))


if __name__ == "__main__":
    unittest.main()
