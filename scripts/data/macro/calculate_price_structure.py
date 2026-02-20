#!/usr/bin/env python3
# -u
# -*- coding: utf-8 -*-

"""
计算个股价格结构指标
- 相对强度
- 动量（累计涨幅）
- 位置（当前价格在历史区间中的位置）
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.macro.paths import DAILY_DIR, PRICE_STRUCTURE_DIR, TUSHARE_ROOT


def calculate_price_structure():
    """计算个股价格结构指标"""
    print("=" * 80)
    print("计算个股价格结构指标")
    print("=" * 80)

    # 加载个股日度数据
    print("\n加载个股日度数据...")
    daily_dir = Path(DAILY_DIR)
    daily_files = sorted(daily_dir.glob("daily_*.parquet"))

    all_daily = []
    for f in daily_files:
        df = pd.read_parquet(f)
        all_daily.append(df)

    daily_df = pd.concat(all_daily, ignore_index=True)
    print(f"  加载 {len(daily_df)} 条记录")
    print(f"  日期范围: {daily_df['trade_date'].min()} ~ {daily_df['trade_date'].max()}")

    # 加载指数数据（沪深300）
    print("\n加载指数数据...")
    index_df = pd.read_parquet(str(TUSHARE_ROOT / "index" / "index_000300_SH_ohlc.parquet"))
    index_df = index_df.rename(columns={"date": "trade_date", "pct_change": "pct_chg"})
    print(f"  加载 {len(index_df)} 条记录")
    print(f"  日期范围: {index_df['trade_date'].min()} ~ {index_df['trade_date'].max()}")

    # 输出文件
    output_dir = Path(PRICE_STRUCTURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "price_structure.parquet"

    # 按股票分组计算指标（向量化）
    print("\n开始计算价格结构指标（向量化）...")

    daily_df["trade_date"] = daily_df["trade_date"].astype(str)
    index_df["trade_date"] = index_df["trade_date"].astype(str)

    daily_df = daily_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    index_df = index_df.sort_values("trade_date").reset_index(drop=True)

    # 动量：使用对数收益滚动求和，避免逐窗口 prod
    daily_df["ret"] = daily_df["pct_chg"] / 100.0
    daily_df["log_ret"] = np.log1p(daily_df["ret"].clip(lower=-0.999999))

    log_ret = daily_df.groupby("ts_code")["log_ret"]
    daily_df["momentum_20d"] = np.expm1(log_ret.rolling(20).sum().reset_index(level=0, drop=True))
    daily_df["momentum_60d"] = np.expm1(log_ret.rolling(60).sum().reset_index(level=0, drop=True))

    # 位置（60日区间位置）
    rolling_min = daily_df.groupby("ts_code")["close"].rolling(60).min().reset_index(level=0, drop=True)
    rolling_max = daily_df.groupby("ts_code")["close"].rolling(60).max().reset_index(level=0, drop=True)
    denom = (rolling_max - rolling_min).replace(0, pd.NA)
    daily_df["position_60d"] = (daily_df["close"] - rolling_min) / denom
    daily_df["position_60d"] = daily_df["position_60d"].fillna(0.5)

    # 指数动量
    index_df["ret"] = index_df["pct_chg"] / 100.0
    index_df["log_ret"] = np.log1p(index_df["ret"].clip(lower=-0.999999))
    index_df["index_momentum_20d"] = np.expm1(index_df["log_ret"].rolling(20).sum())

    # 相对强度 = 个股动量 - 指数动量
    daily_df = daily_df.merge(index_df[["trade_date", "index_momentum_20d"]], on="trade_date", how="left")
    daily_df["relative_strength_20d"] = daily_df["momentum_20d"] - daily_df["index_momentum_20d"]

    final_df = daily_df

    # 保存
    final_df = final_df[
        ["ts_code", "trade_date", "momentum_20d", "momentum_60d", "position_60d", "relative_strength_20d"]
    ]
    final_df.to_parquet(output_file, index=False)
    print(f"  已保存到 {output_file}")

    # 统计
    print("\n" + "=" * 80)
    print("计算完成")
    print("=" * 80)
    print(f"\n总记录数: {len(final_df)}")
    print(f"股票数量: {final_df['ts_code'].nunique()}")
    print(f"日期范围: {final_df['trade_date'].min()} ~ {final_df['trade_date'].max()}")

    # 数据质量检查
    print("\n数据质量检查:")
    for col in ["momentum_20d", "momentum_60d", "position_60d", "relative_strength_20d"]:
        non_null = final_df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(final_df)} ({non_null/len(final_df)*100:.1f}%)")

    print("\n数据预览:")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    calculate_price_structure()
