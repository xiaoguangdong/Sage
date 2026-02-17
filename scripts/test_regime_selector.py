"""
测试分 Regime 选股模型 — 使用真实沪深300成分股数据
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sage_core.stock_selection.regime_stock_selector import (
    RegimeSelectionConfig,
    RegimeStockSelector,
    REGIME_NAMES,
)
from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector
from sage_core.trend.trend_model import TrendModelConfig, TrendModelRuleV2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "tushare")


def load_index_data() -> pd.DataFrame:
    """加载沪深300指数"""
    path = os.path.join(DATA_ROOT, "index", "index_000300_SH_ohlc.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2026-12-31")].reset_index(drop=True)
    print(f"指数数据: {df['date'].min().date()} ~ {df['date'].max().date()}, {len(df)} 条")
    return df


def load_stock_data(years: list) -> pd.DataFrame:
    """加载股票日线数据（多年合并）"""
    frames = []
    for y in years:
        path = os.path.join(DATA_ROOT, "daily", f"daily_{y}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

    # 添加 turnover 列（用 vol/amount 代替）
    if "turnover" not in df.columns and "vol" in df.columns:
        df["turnover"] = df["vol"]

    print(f"股票数据: {len(df)} 条, {df['ts_code'].nunique()} 只股票")
    return df


def load_daily_basic() -> pd.DataFrame:
    """加载每日基本面"""
    path = os.path.join(DATA_ROOT, "daily_basic_all.parquet")
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return df


def load_industry_mapping() -> pd.DataFrame:
    """加载行业映射"""
    path = os.path.join(DATA_ROOT, "sw_industry", "sw_industry_l1.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def merge_data(df_stock: pd.DataFrame, df_basic: pd.DataFrame, df_industry: pd.DataFrame) -> pd.DataFrame:
    """合并股票数据、基本面、行业"""
    # 合并基本面
    basic_cols = ["ts_code", "trade_date", "pe_ttm", "pb", "turnover_rate", "total_mv", "circ_mv"]
    available = [c for c in basic_cols if c in df_basic.columns]
    df = df_stock.merge(df_basic[available], on=["ts_code", "trade_date"], how="left")

    # 合并行业
    if not df_industry.empty and "ts_code" in df_industry.columns:
        ind_col = "industry_name" if "industry_name" in df_industry.columns else "industry_l1"
        if ind_col in df_industry.columns:
            df = df.merge(
                df_industry[["ts_code", ind_col]].drop_duplicates("ts_code"),
                on="ts_code",
                how="left",
            )
            df.rename(columns={ind_col: "industry_l1"}, inplace=True)

    return df


def run_test():
    """主测试流程"""
    print("=" * 60)
    print("分 Regime 选股模型测试")
    print("=" * 60)

    # 1. 加载数据
    df_index = load_index_data()
    df_stock = load_stock_data([2020, 2021, 2022, 2023])
    df_basic = load_daily_basic()
    df_industry = load_industry_mapping()
    df = merge_data(df_stock, df_basic, df_industry)

    # 2. 生成 regime 标签
    print("\n生成 regime 标签...")
    trend_model = TrendModelRuleV2(TrendModelConfig())
    result = trend_model.predict(df_index, return_history=True)
    states = result.diagnostics["states"]
    idx_dates = pd.to_datetime(df_index["date"]).values
    date_to_state = dict(zip(idx_dates, states))

    df["regime"] = df["trade_date"].map(lambda d: date_to_state.get(pd.Timestamp(d), 1))

    # 统计 regime 分布
    regime_counts = df["regime"].value_counts().sort_index()
    for r, cnt in regime_counts.items():
        print(f"  {REGIME_NAMES.get(r, '?')}: {cnt} 条 ({cnt/len(df)*100:.1f}%)")

    # 3. 准备训练/测试集（简单按时间切分）
    split_date = pd.Timestamp("2023-01-01")
    df_train = df[df["trade_date"] < split_date].copy()
    df_test = df[df["trade_date"] >= split_date].copy()
    print(f"\n训练集: {len(df_train)} 条, 测试集: {len(df_test)} 条")

    regime_train = df_train["regime"]

    # 4. 配置
    base_cfg = SelectionConfig(
        model_type="lgbm",
        label_horizons=(10,),
        label_neutralized=True,
        market_neutralized=False,
        ic_filter_enabled=True,
        ic_threshold=0.02,
        ic_ir_threshold=0.3,
        industry_col="industry_l1",
    )

    # 5. 训练分 Regime 模型
    print("\n" + "=" * 60)
    print("训练分 Regime 选股模型")
    print("=" * 60)
    regime_cfg = RegimeSelectionConfig(base_config=base_cfg)
    regime_selector = RegimeStockSelector(regime_cfg)
    regime_selector.fit(df_train, regime_train)

    # 6. 训练单一模型（对照组）
    print("\n" + "=" * 60)
    print("训练单一选股模型（对照组）")
    print("=" * 60)
    single_selector = StockSelector(base_cfg)
    single_selector.fit(df_train)

    # 7. 测试集评估
    print("\n" + "=" * 60)
    print("测试集评估")
    print("=" * 60)

    test_dates = sorted(df_test["trade_date"].unique())
    # 采样部分日期评估
    sample_dates = test_dates[::20][:10]

    for date in sample_dates:
        df_day = df_test[df_test["trade_date"] == date]
        if len(df_day) < 50:
            continue

        regime = date_to_state.get(pd.Timestamp(date), 1)
        regime_name = REGIME_NAMES.get(regime, "?")

        try:
            pred_regime = regime_selector.predict(df_day, regime)
            pred_single = single_selector.predict(df_day)
            print(f"  {str(date)[:10]} [{regime_name}] "
                  f"regime_top10_score={pred_regime['score'].nlargest(10).mean():.4f}, "
                  f"single_top10_score={pred_single['score'].nlargest(10).mean():.4f}")
        except Exception as e:
            print(f"  {str(date)[:10]} [{regime_name}] 预测失败: {e}")

    print("\n✓ 测试完成！")


if __name__ == "__main__":
    run_test()
