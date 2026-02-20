#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试长周期基本面特征计算

用法：
  python scripts/features/test_long_term_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sage_core.features.long_term_fundamental_features import LongTermFundamentalFeatures
from scripts.data._shared.runtime import get_tushare_root


def test_features():
    """测试特征计算"""
    data_root = get_tushare_root()
    calculator = LongTermFundamentalFeatures(data_root)

    # 测试时间范围（使用2024年数据，确保有足够历史数据）
    start_date = "20240101"
    end_date = "20241231"

    print("=" * 60)
    print("测试长周期基本面特征计算")
    print("=" * 60)

    # 测试1：研发费用率
    print("\n【测试1】研发费用率（TTM）")
    try:
        df_rd = calculator.calculate_rd_expense_ratio_ttm(start_date, end_date)
        print(f"✓ 成功计算 {len(df_rd)} 条记录")
        print("\n样例数据（前5条）：")
        print(df_rd.head())
        print("\n统计信息：")
        print(df_rd["rd_expense_ratio_ttm"].describe())
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试2：ROE 5年均值
    print("\n【测试2】ROE 5年均值")
    try:
        df_roe = calculator.calculate_roe_5y_avg(start_date, end_date)
        print(f"✓ 成功计算 {len(df_roe)} 条记录")
        print("\n样例数据（前5条）：")
        print(df_roe.head())
        print("\n统计信息：")
        print(df_roe["roe_5y_avg"].describe())
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试3：营收3年CAGR
    print("\n【测试3】营收3年CAGR")
    try:
        df_revenue_cagr = calculator.calculate_revenue_cagr_3y(start_date, end_date)
        print(f"✓ 成功计算 {len(df_revenue_cagr)} 条记录")
        print("\n样例数据（前5条）：")
        print(df_revenue_cagr.head())
        print("\n统计信息：")
        print(df_revenue_cagr["revenue_cagr_3y"].describe())
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试4：利润3年CAGR
    print("\n【测试4】利润3年CAGR")
    try:
        df_profit_cagr = calculator.calculate_profit_cagr_3y(start_date, end_date)
        print(f"✓ 成功计算 {len(df_profit_cagr)} 条记录")
        print("\n样例数据（前5条）：")
        print(df_profit_cagr.head())
        print("\n统计信息：")
        print(df_profit_cagr["profit_cagr_3y"].describe())
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试5：连续分红年数
    print("\n【测试5】连续分红年数")
    try:
        df_dividend = calculator.calculate_consecutive_dividend_years(start_date, end_date)
        print(f"✓ 成功计算 {len(df_dividend)} 条记录")
        print("\n样例数据（前5条）：")
        print(df_dividend.head())
        print("\n统计信息：")
        print(df_dividend["consecutive_dividend_years"].describe())
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试6：利息保障倍数
    print("\n【测试6】利息保障倍数")
    try:
        df_interest = calculator.calculate_interest_coverage_ratio(start_date, end_date)
        print(f"✓ 成功计算 {len(df_interest)} 条记录")
        print("\n样例数据（前5条）：")
        print(df_interest.head())
        print("\n统计信息：")
        print(df_interest["interest_coverage_ratio"].describe())
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试7：资产负债与净现金
    print("\n【测试7】资产负债与净现金")
    try:
        df_balance = calculator.calculate_balance_sheet_quality_metrics(start_date, end_date)
        print(f"✓ 成功计算 {len(df_balance)} 条记录")
        print("\n样例数据（前5条）：")
        print(df_balance.head())
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试8：现金流质量
    print("\n【测试8】现金流质量（TTM）")
    try:
        df_cashflow = calculator.calculate_cashflow_quality_metrics(start_date, end_date)
        print(f"✓ 成功计算 {len(df_cashflow)} 条记录")
        print("\n样例数据（前5条）：")
        print(df_cashflow.head())
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试9：费用率
    print("\n【测试9】费用率（TTM）")
    try:
        df_expense = calculator.calculate_expense_ratio_ttm(start_date, end_date)
        print(f"✓ 成功计算 {len(df_expense)} 条记录")
        print("\n样例数据（前5条）：")
        print(df_expense.head())
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试10：扣非净利润质量
    print("\n【测试10】扣非净利润质量（TTM）")
    try:
        df_sustainable = calculator.calculate_sustainable_profit_metrics(start_date, end_date)
        print(f"✓ 成功计算 {len(df_sustainable)} 条记录")
        print("\n样例数据（前5条）：")
        print(df_sustainable.head())
    except Exception as e:
        print(f"✗ 失败: {e}")

    # 测试11：计算所有特征
    print("\n【测试11】计算所有特征（合并）")
    try:
        df_all = calculator.calculate_all_features(start_date, end_date)
        print(f"✓ 成功计算 {len(df_all)} 条记录")
        print(f"\n特征列表：{df_all.columns.tolist()}")
        print("\n样例数据（前5条）：")
        print(df_all.head())

        # 检查缺失值
        print("\n缺失值统计：")
        missing = df_all.isnull().sum()
        missing_pct = (missing / len(df_all) * 100).round(2)
        missing_df = pd.DataFrame({"缺失数量": missing, "缺失比例(%)": missing_pct})
        print(missing_df[missing_df["缺失数量"] > 0])

    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_features()
