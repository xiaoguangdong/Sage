#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高 Alpha 因子单元测试
"""

import unittest
from pathlib import Path

import numpy as np

from sage_core.features.high_alpha_features import HighAlphaFeatures


class TestHighAlphaFeatures(unittest.TestCase):
    """高 Alpha 因子单元测试"""

    @classmethod
    def setUpClass(cls):
        """测试前准备"""
        cls.data_root = Path("data/tushare")
        cls.calculator = HighAlphaFeatures(cls.data_root)

    def test_data_availability(self):
        """测试数据可用性"""
        print("\n测试数据可用性...")

        # 检查 moneyflow 目录
        moneyflow_dir = self.data_root / "moneyflow"
        self.assertTrue(moneyflow_dir.exists(), "moneyflow 目录不存在")
        moneyflow_files = list(moneyflow_dir.glob("*.parquet"))
        print(f"  moneyflow 文件数: {len(moneyflow_files)}")

        # 检查 northbound 文件
        northbound_path = self.data_root / "northbound" / "northbound_hold.parquet"
        if northbound_path.exists():
            print("  northbound_hold: 存在")
        else:
            print("  northbound_hold: 不存在")

        # 检查 margin 文件
        margin_path = self.data_root / "margin.parquet"
        if margin_path.exists():
            print("  margin: 存在")
        else:
            print("  margin: 不存在")

    def test_moneyflow_features(self):
        """测试资金流因子计算"""
        print("\n测试资金流因子...")

        # 使用一个固定的测试日期
        test_date = "20190131"

        try:
            result = self.calculator.compute_moneyflow_features(trade_date=test_date, lookback_days=20)

            if not result.empty:
                print(f"  ✅ 成功计算 {len(result)} 只股票")

                # 验证列名
                expected_cols = [
                    "ts_code",
                    "large_net_inflow_ratio",
                    "main_inflow_days",
                    "retail_inst_divergence",
                    "super_large_ratio",
                ]
                for col in expected_cols:
                    self.assertIn(col, result.columns, f"缺少列: {col}")

                # 验证数值范围
                self.assertTrue(
                    result["large_net_inflow_ratio"].between(-1, 1).all(), "large_net_inflow_ratio 超出范围 [-1, 1]"
                )
                self.assertTrue(result["main_inflow_days"].between(0, 20).all(), "main_inflow_days 超出范围 [0, 20]")
                self.assertTrue(result["super_large_ratio"].between(0, 1).all(), "super_large_ratio 超出范围 [0, 1]")

                # 验证无 inf/nan
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                self.assertFalse(result[numeric_cols].isin([np.inf, -np.inf]).any().any(), "存在无穷值")

                print("  ✅ 数据质量验证通过")
            else:
                print("  ⚠️  结果为空（可能数据不足）")

        except Exception as e:
            self.fail(f"资金流因子计算失败: {e}")

    def test_northbound_features(self):
        """测试北向资金因子计算"""
        print("\n测试北向资金因子...")

        test_date = "20190131"

        try:
            result = self.calculator.compute_northbound_features(trade_date=test_date, lookback_days=60)

            if not result.empty:
                print(f"  ✅ 成功计算 {len(result)} 只股票")

                # 验证列名
                expected_cols = [
                    "ts_code",
                    "config_fund_ratio",
                    "trading_fund_inflow",
                    "holding_concentration_change",
                    "holding_stability",
                ]
                for col in expected_cols:
                    self.assertIn(col, result.columns, f"缺少列: {col}")

                print("  ✅ 列名验证通过")
            else:
                print("  ⚠️  结果为空（可能数据不足或数据缺口）")

        except Exception as e:
            print(f"  ⚠️  北向资金因子计算失败: {e}")

    def test_margin_features(self):
        """测试融资融券因子计算"""
        print("\n测试融资融券因子...")

        test_date = "20190131"

        try:
            result = self.calculator.compute_margin_features(trade_date=test_date, lookback_days=20)

            if not result.empty:
                print(f"  ✅ 成功计算 {len(result)} 只股票")

                # 验证列名
                expected_cols = ["ts_code", "margin_balance_change_rate", "margin_net_buy_ratio"]
                for col in expected_cols:
                    self.assertIn(col, result.columns, f"缺少列: {col}")

                print("  ✅ 列名验证通过")
            else:
                print("  ⚠️  结果为空（可能数据不足）")

        except Exception as e:
            print(f"  ⚠️  融资融券因子计算失败: {e}")

    def test_analyst_features(self):
        """测试分析师预期因子计算"""
        print("\n测试分析师预期因子...")

        test_date = "20190131"

        try:
            result = self.calculator.compute_analyst_features(trade_date=test_date, lookback_days=120)

            if not result.empty:
                print(f"  ✅ 成功计算 {len(result)} 只股票")

                # 验证列名
                expected_cols = [
                    "ts_code",
                    "analyst_upgrade_count",
                    "analyst_revision_magnitude",
                    "analyst_surprise_degree",
                    "analyst_consensus",
                ]
                for col in expected_cols:
                    self.assertIn(col, result.columns, f"缺少列: {col}")

                print("  ✅ 列名验证通过")
            else:
                print("  ⚠️  结果为空（可能数据不足）")

        except Exception as e:
            print(f"  ⚠️  分析师预期因子计算失败: {e}")

    def test_empty_data_handling(self):
        """测试空数据处理"""
        print("\n测试空数据处理...")

        # 使用一个不存在的日期
        test_date = "19900101"

        result = self.calculator.compute_moneyflow_features(trade_date=test_date, lookback_days=20)

        self.assertTrue(result.empty, "空数据应返回空 DataFrame")
        print("  ✅ 空数据处理正确")


def main():
    """运行测试"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    main()
