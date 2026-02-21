"""
选股模型回测 — 2024-09-10 ~ 2026-01-01
对比：单一模型 / 分Regime模型 / 沪深300基准
持仓周期：20个交易日
"""

import copy
import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ConstantInput.*")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sage_core.stock_selection.regime_stock_selector import REGIME_NAMES, RegimeSelectionConfig, RegimeStockSelector
from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector
from sage_core.trend.trend_model import TrendModelConfig, TrendModelRuleV2
from scripts.data._shared.runtime import log_task_summary, setup_logger

logger = setup_logger("backtest_stock_selection", module="backtest")

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "tushare")

# ── 回测参数 ──
BT_START = "2024-09-10"
BT_END = "2026-01-01"
TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
HOLD_DAYS = 20
TOP_N = 30


def load_index_data() -> pd.DataFrame:
    path = os.path.join(DATA_ROOT, "index", "index_000300_SH_ohlc.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_stock_data(years: list) -> pd.DataFrame:
    frames = []
    for y in years:
        path = os.path.join(DATA_ROOT, "daily", f"daily_{y}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    if "turnover" not in df.columns and "vol" in df.columns:
        df["turnover"] = df["vol"]
    return df


def load_daily_basic() -> pd.DataFrame:
    path = os.path.join(DATA_ROOT, "daily_basic_all.parquet")
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return df


def load_industry_mapping() -> pd.DataFrame:
    path = os.path.join(DATA_ROOT, "sw_industry", "sw_industry_l1.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def merge_data(df_stock, df_basic, df_industry):
    basic_cols = ["ts_code", "trade_date", "pe_ttm", "pb", "turnover_rate", "total_mv", "circ_mv"]
    available = [c for c in basic_cols if c in df_basic.columns]
    df = df_stock.merge(df_basic[available], on=["ts_code", "trade_date"], how="left")
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


def compute_portfolio_return(df_all, selected_codes, start_date, end_date):
    """计算持仓期间等权组合收益"""
    port = df_all[
        (df_all["ts_code"].isin(selected_codes))
        & (df_all["trade_date"] >= start_date)
        & (df_all["trade_date"] <= end_date)
    ].copy()
    if port.empty:
        return pd.DataFrame()
    port.groupby("trade_date").apply(lambda g: g["close"].values / g.groupby("ts_code")["close"].shift(1).values - 1)
    # 简化：用每日截面平均收益
    port["ret"] = port.groupby("ts_code")["close"].pct_change()
    daily_avg = port.groupby("trade_date")["ret"].mean().dropna()
    return daily_avg


def run_backtest():
    print("=" * 60)
    print(f"选股模型回测: {BT_START} ~ {BT_END}")
    print(f"持仓周期: {HOLD_DAYS}天, Top {TOP_N}")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    df_index = load_index_data()
    df_stock = load_stock_data(TRAIN_YEARS + [2025, 2026])
    df_basic = load_daily_basic()
    df_industry = load_industry_mapping()
    df_all = merge_data(df_stock, df_basic, df_industry)
    print(f"  股票数据: {len(df_all)} 条, {df_all['ts_code'].nunique()} 只")

    # 训练数据（回测开始前）
    train_end = pd.Timestamp(BT_START)
    df_train = df_all[df_all["trade_date"] < train_end].copy()
    print(f"  训练集: {len(df_train)} 条")

    # 生成 regime 标签
    trend_model = TrendModelRuleV2(TrendModelConfig())
    idx_train = df_index[df_index["date"] < train_end].copy()
    result = trend_model.predict(idx_train, return_history=True)
    states = result.diagnostics["states"]
    idx_dates = pd.to_datetime(idx_train["date"]).values
    date_to_state = dict(zip(idx_dates, states))
    df_train["regime"] = df_train["trade_date"].map(lambda d: date_to_state.get(pd.Timestamp(d), 1))

    # 配置
    base_cfg = SelectionConfig(
        model_type="lgbm",
        label_neutralized=True,
        ic_filter_enabled=True,
        ic_threshold=0.02,
        ic_ir_threshold=0.3,
        industry_col="industry_l1",
    )

    # ── 训练模型 ──
    print("\n训练单一模型...")
    single_model = StockSelector(copy.deepcopy(base_cfg))
    single_model.fit(df_train)

    print("训练分Regime模型...")
    regime_cfg = RegimeSelectionConfig(base_config=copy.deepcopy(base_cfg))
    regime_model = RegimeStockSelector(regime_cfg)
    regime_model.fit(df_train, df_train["regime"])

    # 预计算全量特征（修复：避免单日数据导致时序特征全NaN）
    print("预计算全量特征...")
    df_prepared = single_model.prepare_features(df_all)

    # ── 回测 ──
    print(f"\n开始回测 {BT_START} ~ {BT_END}...")
    bt_dates = sorted(
        df_all[(df_all["trade_date"] >= BT_START) & (df_all["trade_date"] <= BT_END)]["trade_date"].unique()
    )

    # 每 HOLD_DAYS 天调仓
    rebalance_dates = bt_dates[::HOLD_DAYS]

    # 获取回测期间的 regime
    idx_bt = df_index[df_index["date"] <= BT_END].copy()
    result_bt = trend_model.predict(idx_bt, return_history=True)
    states_bt = result_bt.diagnostics["states"]
    idx_dates_bt = pd.to_datetime(idx_bt["date"]).values
    date_to_state_bt = dict(zip(idx_dates_bt, states_bt))

    records_single = []
    records_regime = []

    for i, rb_date in enumerate(rebalance_dates):
        hold_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else bt_dates[-1]
        df_day = df_prepared[df_prepared["trade_date"] == rb_date]
        if len(df_day) < 50:
            continue

        regime = date_to_state_bt.get(pd.Timestamp(rb_date), 1)
        regime_name = REGIME_NAMES.get(regime, "?")

        try:
            pred_s = single_model.predict_prepared(df_day)
            top_s = pred_s.nlargest(TOP_N, "score")["ts_code"].tolist()

            pred_r = regime_model.predict_prepared(df_day, regime)
            top_r = pred_r.nlargest(TOP_N, "score")["ts_code"].tolist()

            # 计算持仓期收益
            hold_data = df_all[(df_all["trade_date"] > rb_date) & (df_all["trade_date"] <= hold_end)]

            for code_list, records in [(top_s, records_single), (top_r, records_regime)]:
                port = hold_data[hold_data["ts_code"].isin(code_list)]
                port.groupby("trade_date")["close"].apply(lambda s: s.pct_change().mean() if len(s) > 1 else 0)
                # 用更准确的方式：每只股票的持仓期收益
                stock_rets = []
                for code in code_list:
                    stk = hold_data[hold_data["ts_code"] == code].sort_values("trade_date")
                    if len(stk) >= 2:
                        ret = stk["close"].iloc[-1] / stk["close"].iloc[0] - 1
                        stock_rets.append(ret)
                period_ret = np.mean(stock_rets) if stock_rets else 0
                records.append(
                    {
                        "rebalance_date": rb_date,
                        "hold_end": hold_end,
                        "regime": regime_name,
                        "period_return": period_ret,
                        "n_stocks": len(stock_rets),
                    }
                )

            print(
                f"  {str(rb_date)[:10]} [{regime_name:7s}] "
                f"单一={records_single[-1]['period_return']:+.2%}, "
                f"Regime={records_regime[-1]['period_return']:+.2%}"
            )

        except Exception as e:
            print(f"  {str(rb_date)[:10]} 失败: {e}")

    # ── 汇总 ──
    df_single = pd.DataFrame(records_single)
    df_regime = pd.DataFrame(records_regime)

    # 沪深300收益
    idx_bt_period = df_index[(df_index["date"] >= BT_START) & (df_index["date"] <= BT_END)].sort_values("date")

    # 计算累计净值
    def cum_nav(period_returns):
        nav = [1.0]
        for r in period_returns:
            nav.append(nav[-1] * (1 + r))
        return nav

    nav_single = cum_nav(df_single["period_return"].tolist())
    nav_regime = cum_nav(df_regime["period_return"].tolist())

    idx_start_price = idx_bt_period["close"].iloc[0]
    (idx_bt_period["close"] / idx_start_price).values

    # 在调仓日对齐指数净值
    nav_dates = [pd.Timestamp(BT_START)] + df_single["hold_end"].tolist()
    idx_nav_at_rb = []
    for d in nav_dates:
        mask = idx_bt_period["date"] <= d
        if mask.any():
            idx_nav_at_rb.append(idx_bt_period[mask]["close"].iloc[-1] / idx_start_price)
        else:
            idx_nav_at_rb.append(1.0)

    # ── 统计 ──
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)

    for name, df_r, nav in [("单一模型", df_single, nav_single), ("Regime模型", df_regime, nav_regime)]:
        total_ret = nav[-1] - 1
        n_periods = len(df_r)
        win_rate = (df_r["period_return"] > 0).mean()
        avg_ret = df_r["period_return"].mean()
        std_ret = df_r["period_return"].std()
        sharpe = avg_ret / (std_ret + 1e-8) * np.sqrt(252 / HOLD_DAYS)
        max_dd = min(np.array(nav) / np.maximum.accumulate(nav) - 1)

        print(f"\n{name}:")
        print(f"  累计收益: {total_ret:+.2%}")
        print(f"  年化Sharpe: {sharpe:.2f}")
        print(f"  最大回撤: {max_dd:.2%}")
        print(f"  胜率: {win_rate:.1%} ({n_periods}期)")
        print(f"  平均每期收益: {avg_ret:+.2%}")

    idx_total = idx_nav_at_rb[-1] - 1
    print("\n沪深300:")
    print(f"  累计收益: {idx_total:+.2%}")

    # 按 regime 分析
    print("\n" + "-" * 40)
    print("分 Regime 表现:")
    for regime_name in ["bull", "neutral", "bear"]:
        mask_s = df_single["regime"] == regime_name
        mask_r = df_regime["regime"] == regime_name
        if mask_s.sum() > 0:
            print(
                f"  [{regime_name:7s}] 单一={df_single[mask_s]['period_return'].mean():+.2%}, "
                f"Regime={df_regime[mask_r]['period_return'].mean():+.2%}, "
                f"期数={mask_s.sum()}"
            )

    # ── 可视化 ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # 净值曲线
    ax = axes[0]
    ax.plot(nav_dates, nav_single, "b-o", markersize=4, label=f"单一模型 ({nav_single[-1]-1:+.1%})", linewidth=2)
    ax.plot(nav_dates, nav_regime, "r-s", markersize=4, label=f"Regime模型 ({nav_regime[-1]-1:+.1%})", linewidth=2)
    ax.plot(nav_dates, idx_nav_at_rb, "k--", label=f"沪深300 ({idx_nav_at_rb[-1]-1:+.1%})", linewidth=1.5, alpha=0.7)

    # 标记 regime 背景色
    for i, row in df_single.iterrows():
        color = {"bull": "green", "neutral": "gray", "bear": "red"}.get(row["regime"], "white")
        ax.axvspan(row["rebalance_date"], row["hold_end"], alpha=0.08, color=color)

    ax.set_ylabel("累计净值", fontsize=12)
    ax.set_title(f"选股模型回测 {BT_START} ~ {BT_END} (Top{TOP_N}, {HOLD_DAYS}天调仓)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="gray", linestyle="-", linewidth=0.5)

    # 每期超额收益
    ax2 = axes[1]
    x = range(len(df_single))
    df_single["period_return"].values - np.diff([1.0] + idx_nav_at_rb[1:])
    df_regime["period_return"].values - np.diff([1.0] + idx_nav_at_rb[1:])

    # 简化：直接用绝对收益柱状图
    width = 0.35
    ax2.bar(
        [i - width / 2 for i in x],
        df_single["period_return"].values * 100,
        width,
        label="单一模型",
        color="steelblue",
        alpha=0.7,
    )
    ax2.bar(
        [i + width / 2 for i in x],
        df_regime["period_return"].values * 100,
        width,
        label="Regime模型",
        color="indianred",
        alpha=0.7,
    )
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("每期收益 (%)", fontsize=11)
    ax2.set_xlabel("调仓期", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 添加 regime 标签
    for i, row in df_single.iterrows():
        ax2.text(i, ax2.get_ylim()[0] * 0.9, row["regime"][:1].upper(), ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    output_dir = "images/backtest"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "stock_selection_backtest_2024_2026.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存: {output_file}")
    plt.close()

    print("\n✓ 回测完成！")


if __name__ == "__main__":
    start_time = datetime.now().timestamp()
    failure_reason = None
    try:
        run_backtest()
    except Exception as exc:
        failure_reason = str(exc)
        raise
    finally:
        log_task_summary(
            logger,
            task_name="backtest_stock_selection",
            window=f"{BT_START}~{BT_END}",
            elapsed_s=datetime.now().timestamp() - start_time,
            error=failure_reason,
        )
