#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高 Alpha 因子测试脚本

功能：
1. 加载最近1个月的股票数据
2. 计算高 Alpha 因子
3. 输出因子统计信息（均值、标准差、覆盖率）
4. 保存结果到 CSV 供人工检查
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from sage_core.features.high_alpha_features import HighAlphaFeatures


def test_high_alpha_factors(data_root: Path = Path("data/tushare"), test_date: str = None, lookback_days: int = 20):
    """测试高 Alpha 因子计算

    Args:
        data_root: 数据根目录
        test_date: 测试日期（YYYYMMDD），默认为最近的交易日
        lookback_days: 回溯天数
    """
    print("=" * 60)
    print("高 Alpha 因子测试")
    print("=" * 60)

    calculator = HighAlphaFeatures(data_root)

    # 如果未指定测试日期，使用最近的交易日
    if test_date is None:
        # 从 daily 数据中获取最近的交易日
        daily_files = list((data_root / "daily").glob("*.parquet"))
        if daily_files:
            sample_df = pd.read_parquet(daily_files[0])
            test_date = sample_df["trade_date"].max()
            print(f"\n使用最近交易日: {test_date}")
        else:
            # 默认使用今天
            test_date = datetime.now().strftime("%Y%m%d")
            print(f"\n使用默认日期: {test_date}")

    results = {}

    # 1. 测试资金流因子
    print("\n" + "-" * 60)
    print("1. 测试资金流因子")
    print("-" * 60)
    try:
        moneyflow_feat = calculator.compute_moneyflow_features(trade_date=test_date, lookback_days=lookback_days)
        if not moneyflow_feat.empty:
            print(f"✅ 成功计算 {len(moneyflow_feat)} 只股票的资金流因子")
            print("\n因子统计:")
            for col in moneyflow_feat.columns:
                if col != "ts_code":
                    print(f"  {col}:")
                    print(f"    均值: {moneyflow_feat[col].mean():.4f}")
                    print(f"    标准差: {moneyflow_feat[col].std():.4f}")
                    print(f"    覆盖率: {moneyflow_feat[col].notna().sum() / len(moneyflow_feat) * 100:.1f}%")
            results["moneyflow"] = moneyflow_feat
        else:
            print("⚠️  资金流因子计算结果为空")
    except Exception as e:
        print(f"❌ 资金流因子计算失败: {e}")

    # 2. 测试北向资金因子
    print("\n" + "-" * 60)
    print("2. 测试北向资金因子")
    print("-" * 60)
    try:
        northbound_feat = calculator.compute_northbound_features(trade_date=test_date, lookback_days=lookback_days * 3)
        if not northbound_feat.empty:
            print(f"✅ 成功计算 {len(northbound_feat)} 只股票的北向资金因子")
            print("\n因子统计:")
            for col in northbound_feat.columns:
                if col != "ts_code":
                    print(f"  {col}:")
                    print(f"    均值: {northbound_feat[col].mean():.4f}")
                    print(f"    标准差: {northbound_feat[col].std():.4f}")
                    print(f"    覆盖率: {northbound_feat[col].notna().sum() / len(northbound_feat) * 100:.1f}%")
            results["northbound"] = northbound_feat
        else:
            print("⚠️  北向资金因子计算结果为空")
    except Exception as e:
        print(f"❌ 北向资金因子计算失败: {e}")

    # 3. 测试融资融券因子
    print("\n" + "-" * 60)
    print("3. 测试融资融券因子")
    print("-" * 60)
    try:
        margin_feat = calculator.compute_margin_features(trade_date=test_date, lookback_days=lookback_days)
        if not margin_feat.empty:
            print(f"✅ 成功计算 {len(margin_feat)} 只股票的融资融券因子")
            print("\n因子统计:")
            for col in margin_feat.columns:
                if col != "ts_code":
                    print(f"  {col}:")
                    print(f"    均值: {margin_feat[col].mean():.4f}")
                    print(f"    标准差: {margin_feat[col].std():.4f}")
                    print(f"    覆盖率: {margin_feat[col].notna().sum() / len(margin_feat) * 100:.1f}%")
            results["margin"] = margin_feat
        else:
            print("⚠️  融资融券因子计算结果为空")
    except Exception as e:
        print(f"❌ 融资融券因子计算失败: {e}")

    # 4. 测试分析师预期因子
    print("\n" + "-" * 60)
    print("4. 测试分析师预期因子")
    print("-" * 60)
    try:
        analyst_feat = calculator.compute_analyst_features(trade_date=test_date, lookback_days=lookback_days * 6)
        if not analyst_feat.empty:
            print(f"✅ 成功计算 {len(analyst_feat)} 只股票的分析师预期因子")
            print("\n因子统计:")
            for col in analyst_feat.columns:
                if col != "ts_code":
                    print(f"  {col}:")
                    print(f"    均值: {analyst_feat[col].mean():.4f}")
                    print(f"    标准差: {analyst_feat[col].std():.4f}")
                    print(f"    覆盖率: {analyst_feat[col].notna().sum() / len(analyst_feat) * 100:.1f}%")
            results["analyst"] = analyst_feat
        else:
            print("⚠️  分析师预期因子计算结果为空")
    except Exception as e:
        print(f"❌ 分析师预期因子计算失败: {e}")

    # 保存结果
    if results:
        output_dir = Path("logs/features")
        output_dir.mkdir(parents=True, exist_ok=True)

        for feat_type, feat_df in results.items():
            output_path = output_dir / f"high_alpha_{feat_type}_{test_date}.csv"
            feat_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"\n✅ {feat_type} 因子已保存到: {output_path}")

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"测试日期: {test_date}")
    print(f"回溯天数: {lookback_days}")
    print(f"成功计算的因子组: {len(results)}/4")

    if results:
        # 合并所有因子
        all_factors = None
        for feat_type, feat_df in results.items():
            if all_factors is None:
                all_factors = feat_df
            else:
                all_factors = all_factors.merge(feat_df, on="ts_code", how="outer")

        print(f"\n总覆盖股票数: {len(all_factors)}")
        print(f"总因子数: {len(all_factors.columns) - 1}")

        # 保存合并结果
        merged_path = output_dir / f"high_alpha_all_{test_date}.csv"
        all_factors.to_csv(merged_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ 合并因子已保存到: {merged_path}")

    print("\n" + "=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="测试高 Alpha 因子计算")
    parser.add_argument("--data-root", type=str, default="data/tushare", help="数据根目录")
    parser.add_argument("--test-date", type=str, default=None, help="测试日期（YYYYMMDD）")
    parser.add_argument("--lookback", type=int, default=20, help="回溯天数")

    args = parser.parse_args()

    test_high_alpha_factors(data_root=Path(args.data_root), test_date=args.test_date, lookback_days=args.lookback)


if __name__ == "__main__":
    main()
