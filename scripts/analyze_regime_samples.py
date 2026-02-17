"""分析Regime样本分布"""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sage_core.trend.trend_model import TrendModelConfig, TrendModelRuleV2

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "tushare")
TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]

def load_data():
    idx = pd.read_parquet(os.path.join(DATA_ROOT, "index", "index_000300_SH_ohlc.parquet"))
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)

    frames = []
    for y in TRAIN_YEARS:
        p = os.path.join(DATA_ROOT, "daily", f"daily_{y}.parquet")
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    stk = pd.concat(frames, ignore_index=True)
    stk["trade_date"] = pd.to_datetime(stk["trade_date"], format="%Y%m%d")

    return idx, stk

def main():
    print("加载数据...")
    idx, stk = load_data()

    # 用趋势模型打标签
    cfg = TrendModelConfig(confirmation_periods=3, exit_tolerance=5, min_hold_periods=7)
    trend = TrendModelRuleV2(cfg)

    train_end = pd.Timestamp("2024-09-01")
    idx_train = idx[idx["date"] < train_end].copy()
    res = trend.predict(idx_train, return_history=True)

    # 构建日期到状态的映射
    d2s = dict(zip(pd.to_datetime(idx_train["date"]).values, res.diagnostics["states"]))

    # 统计每日状态
    dates = sorted(d2s.keys())
    states = [d2s[d] for d in dates]

    df_states = pd.DataFrame({"date": dates, "state": states})
    df_states["year"] = df_states["date"].dt.year
    df_states["month"] = df_states["date"].dt.to_period("M")

    print(f"\n{'='*70}")
    print("训练期间状态分布 (2020-2024)")
    print(f"{'='*70}")

    # 总体分布
    total = len(df_states)
    for state, name in [(0, "bear"), (1, "neutral"), (2, "bull")]:
        cnt = (df_states["state"] == state).sum()
        print(f"{name:8s}: {cnt:4d}天 ({cnt/total*100:5.1f}%)")

    print(f"\n{'='*70}")
    print("按年份分布")
    print(f"{'='*70}")
    print(f"{'年份':<8} {'bear':>8} {'neutral':>8} {'bull':>8} {'总天数':>8}")
    print("-" * 70)

    for year in TRAIN_YEARS:
        if year >= 2025:
            continue
        year_data = df_states[df_states["year"] == year]
        if len(year_data) == 0:
            continue
        bear = (year_data["state"] == 0).sum()
        neutral = (year_data["state"] == 1).sum()
        bull = (year_data["state"] == 2).sum()
        total_days = len(year_data)
        print(f"{year:<8} {bear:>8} {neutral:>8} {bull:>8} {total_days:>8}")

    print(f"\n{'='*70}")
    print("最近N个月的样本量分析")
    print(f"{'='*70}")

    for months in [6, 12, 24]:
        cutoff = train_end - pd.DateOffset(months=months)
        recent = df_states[df_states["date"] >= cutoff]

        bear = (recent["state"] == 0).sum()
        neutral = (recent["state"] == 1).sum()
        bull = (recent["state"] == 2).sum()

        print(f"\n最近{months}个月:")
        print(f"  bear: {bear}天, neutral: {neutral}天, bull: {bull}天")

        # 估算股票样本量（假设每天3000只股票）
        print(f"  估算样本量: bear≈{bear*3000:,}, neutral≈{neutral*3000:,}, bull≈{bull*3000:,}")

        # 评估是否足够
        min_samples = 50000  # LightGBM建议最小样本量
        print(f"  是否足够(>{min_samples:,}): ", end="")
        print(f"bear={'✓' if bear*3000>min_samples else '✗'}, ", end="")
        print(f"neutral={'✓' if neutral*3000>min_samples else '✗'}, ", end="")
        print(f"bull={'✓' if bull*3000>min_samples else '✗'}")

if __name__ == "__main__":
    main()
