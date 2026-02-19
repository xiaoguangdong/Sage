"""
测试风险控制模块
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sage_core.portfolio.risk_control import RiskControl


class TestRiskControl(unittest.TestCase):
    """测试风险控制模块"""

    def test_compute_target_position_with_shock_and_drawdown(self):
        """测试冲击+回撤场景下的目标仓位"""
        risk_control = RiskControl()
        result = risk_control.compute_target_position(
            trend_state=0,
            market_volatility=0.20,
            latest_index_return=-0.03,
            latest_week_return=-0.06,
            portfolio_drawdown=-0.13,
        )

        self.assertAlmostEqual(result["base_position"], 0.30, places=6)
        self.assertAlmostEqual(result["position_after_shock"], 0.135, places=6)
        self.assertAlmostEqual(result["final_position"], 0.054, places=6)

    def test_scale_to_target_exposure(self):
        """测试组合缩放到目标仓位"""
        risk_control = RiskControl()
        df_portfolio = pd.DataFrame({"weight": [0.2, 0.3]})
        adjusted = risk_control.scale_to_target_exposure(df_portfolio, 0.25)

        self.assertAlmostEqual(float(adjusted["weight"].sum()), 0.25, places=6)

    def test_adjust_weights_keep_single_position_cap(self):
        """测试保留总仓位时不突破单票上限"""
        risk_control = RiskControl(
            {
                "max_single_position": 0.30,
                "enable_position_limit": True,
                "preserve_gross_exposure": True,
            }
        )
        df_portfolio = pd.DataFrame(
            {
                "code": ["A", "B", "C"],
                "weight": [0.4, 0.4, 0.2],
            }
        )
        adjusted = risk_control.adjust_weights(df_portfolio)

        self.assertTrue((adjusted["weight"] <= 0.30 + 1e-12).all())
        self.assertAlmostEqual(float(adjusted["weight"].sum()), 0.90, places=6)

    def test_compute_market_volatility_from_close(self):
        """测试从close列计算市场波动率"""
        np.random.seed(7)
        close = 3000 * np.cumprod(1 + np.random.normal(0, 0.01, 40))
        df_index = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-01", periods=40, freq="D").strftime("%Y%m%d"),
                "close": close,
            }
        )
        risk_control = RiskControl({"market_vol_window": 20})
        market_vol = risk_control.compute_market_volatility(df_index)

        self.assertTrue(np.isfinite(market_vol))
        self.assertGreater(market_vol, 0.0)


if __name__ == "__main__":
    unittest.main()
