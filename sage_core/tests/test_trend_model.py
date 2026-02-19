"""
测试趋势模型
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# 导入被测试的模块
sys.path.append(str(Path(__file__).resolve().parents[2]))

from sage_core.trend.trend_model import TrendModelRule, create_trend_model


class TestTrendModelRule(unittest.TestCase):
    """测试规则趋势模型"""

    def test_create_trend_model(self):
        """测试创建趋势模型"""
        model = create_trend_model("rule")
        self.assertIsNotNone(model)
        self.assertIsInstance(model, TrendModelRule)

    def test_predict_risk_on(self):
        """测试预测RISK_ON状态"""
        np.random.seed(42)

        # 创建上升趋势+低波动数据
        dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
        close = np.linspace(2800, 3200, len(dates))  # 上升趋势
        noise = np.random.normal(0, 5, len(dates))  # 低波动
        close = close + noise

        df = pd.DataFrame({"date": dates, "close": close})

        model = TrendModelRule()
        result = model.predict(df)

        self.assertEqual(result["state"], 2)
        self.assertEqual(result["state_name"], "RISK_ON")
        self.assertTrue(result["is_up_trend"])
        self.assertTrue(result["is_low_vol"])

    def test_predict_risk_off(self):
        """测试预测RISK_OFF状态"""
        np.random.seed(42)

        # 创建下降趋势数据
        dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
        close = np.linspace(3200, 2800, len(dates))  # 下降趋势
        noise = np.random.normal(0, 20, len(dates))  # 高波动
        close = close + noise

        df = pd.DataFrame({"date": dates, "close": close})

        model = TrendModelRule()
        result = model.predict(df)

        self.assertEqual(result["state"], 0)
        self.assertEqual(result["state_name"], "RISK_OFF")
        self.assertFalse(result["is_up_trend"])

    def test_predict_neutral(self):
        """测试预测震荡状态"""
        np.random.seed(42)

        # 创建横盘数据
        dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
        close = 3000 + np.random.randn(len(dates)) * 10  # 横盘

        df = pd.DataFrame({"date": dates, "close": close})

        model = TrendModelRule()
        result = model.predict(df)

        # 横盘可能被判断为震荡
        self.assertIn(result["state"], [0, 1])
        self.assertIn(result["state_name"], ["RISK_OFF", "震荡"])

    def test_missing_close_column(self):
        """测试缺少close列"""
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", "2020-01-10")})

        model = TrendModelRule()

        with self.assertRaises(ValueError):
            model.predict(df)


if __name__ == "__main__":
    unittest.main()
