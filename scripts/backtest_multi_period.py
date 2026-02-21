"""选股模型多区间回测 — 支持趋势模型仓位联动 + 增强风控"""

import copy
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sage_core.backtest.cost_model import create_default_cost_model
from sage_core.portfolio.enhanced_risk_control import EnhancedRiskControl, RiskControlConfig
from sage_core.stock_selection.regime_stock_selector import REGIME_NAMES, RegimeSelectionConfig, RegimeStockSelector
from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector
from sage_core.trend.trend_model import TrendModelConfig, TrendModelRuleV2
from scripts.data._shared.runtime import log_task_summary, setup_logger

logger = setup_logger("backtest_multi_period", module="backtest")
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "tushare")
HOLD_DAYS, TOP_N = 20, 30

# 趋势联动仓位
REGIME_POSITION = {0: 0.3, 1: 0.6, 2: 1.0}  # bear/neutral/bull


def build_ma_regime(idx: pd.DataFrame) -> dict:
    """用 MA20/MA60 均线判断中期 regime（选股用，比趋势模型更稳定）"""
    df = idx.sort_values("date").copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    regime = np.where(
        (df["close"] > df["ma60"]) & (df["ma20"] > df["ma60"]),
        2,  # bull
        np.where((df["close"] < df["ma60"]) & (df["ma20"] < df["ma60"]), 0, 1),  # bear  # neutral
    )
    return dict(zip(pd.to_datetime(df["date"]).values, regime))


# 宏观级别 regime 标签（基于沪深300实际走势人工划分）
MACRO_REGIMES = [
    ("2020-01-01", "2021-02-28", 2),  # bull +28.5%
    ("2021-03-01", "2023-12-31", 0),  # bear -36.7%
    ("2024-01-01", "2024-08-31", 1),  # neutral -1.9%
    ("2024-09-01", "2026-12-31", 2),  # bull +41.8%
]


def assign_macro_regime(dates: pd.Series) -> pd.Series:
    """根据宏观日期区间分配 regime 标签"""
    regime = pd.Series(1, index=dates.index)  # 默认 neutral
    for start, end, r in MACRO_REGIMES:
        mask = (dates >= start) & (dates <= end)
        regime[mask] = r
    return regime


def load_all_data():
    idx = pd.read_parquet(os.path.join(DATA_ROOT, "index", "index_000300_SH_ohlc.parquet"))
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)

    frames = []
    for y in range(2020, 2027):
        p = os.path.join(DATA_ROOT, "daily", f"daily_{y}.parquet")
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    stk = pd.concat(frames, ignore_index=True)
    stk["trade_date"] = pd.to_datetime(stk["trade_date"], format="%Y%m%d")
    if "turnover" not in stk.columns and "vol" in stk.columns:
        stk["turnover"] = stk["vol"]

    basic = pd.read_parquet(os.path.join(DATA_ROOT, "daily_basic_all.parquet"))
    basic["trade_date"] = pd.to_datetime(basic["trade_date"], format="%Y%m%d")
    cols = [
        c
        for c in ["ts_code", "trade_date", "pe_ttm", "pb", "turnover_rate", "total_mv", "circ_mv"]
        if c in basic.columns
    ]
    df = stk.merge(basic[cols], on=["ts_code", "trade_date"], how="left")

    member_path = os.path.join(DATA_ROOT, "sw_industry", "sw_index_member.parquet")
    ind_path = os.path.join(DATA_ROOT, "sw_industry", "sw_industry_l1.parquet")
    if os.path.exists(member_path) and os.path.exists(ind_path):
        member = pd.read_parquet(member_path)
        ind = pd.read_parquet(ind_path)
        # 只取当前有效成员
        current = member[member["is_new"] == "Y"][["index_code", "con_code"]].drop_duplicates("con_code")
        current = current.merge(ind[["index_code", "industry_name"]], on="index_code", how="left")
        df = df.merge(
            current[["con_code", "industry_name"]].rename(
                columns={"con_code": "ts_code", "industry_name": "industry_l1"}
            ),
            on="ts_code",
            how="left",
        )

    # ST 标记（用于股票池过滤）
    ths_path = os.path.join(DATA_ROOT, "concepts", "ths_member.parquet")
    if os.path.exists(ths_path):
        ths = pd.read_parquet(ths_path)
        st_codes = set(ths[ths["con_name"].str.contains("ST", na=False)]["con_code"].unique())
        df["is_st"] = df["ts_code"].isin(st_codes)

    return idx, df


def _compute_period_return_with_stop_loss(df_all, codes, rb, hold_end, entry_prices_cache, risk_ctrl):
    """计算持仓期收益，支持个股ATR止损（日频模拟）

    Args:
        df_all: 全量日线数据
        codes: 持仓股票列表
        rb: 调仓日
        hold_end: 持仓结束日
        entry_prices_cache: 入场价缓存（会被更新）
        risk_ctrl: EnhancedRiskControl 实例

    Returns:
        (period_return, n_stocks, n_stopped): 期间收益、持仓数、止损数
    """
    hold = df_all[(df_all["trade_date"] > rb) & (df_all["trade_date"] <= hold_end)]
    if hold.empty:
        return 0.0, 0, 0

    hold_dates = sorted(hold["trade_date"].unique())
    active_codes = set(codes)
    stopped_codes = set()

    # 记录入场价（调仓日收盘价）
    rb_data = df_all[df_all["trade_date"] == rb]
    for c in codes:
        stk = rb_data[rb_data["ts_code"] == c]
        if not stk.empty:
            entry_prices_cache[c] = float(stk["close"].iloc[0])

    # 日频模拟：逐日检查止损
    # 累积每只股票的净值
    stock_nav = {c: 1.0 for c in codes}
    prev_close = {}
    for c in codes:
        stk = rb_data[rb_data["ts_code"] == c]
        if not stk.empty:
            prev_close[c] = float(stk["close"].iloc[0])

    for day in hold_dates:
        day_data = hold[hold["trade_date"] == day]
        for c in list(active_codes):
            stk = day_data[day_data["ts_code"] == c]
            if stk.empty:
                continue
            cur_close = float(stk["close"].iloc[0])
            entry_price = entry_prices_cache.get(c)
            if entry_price is None or entry_price <= 0:
                prev_close[c] = cur_close
                continue

            # 更新净值
            if c in prev_close and prev_close[c] > 0:
                stock_nav[c] *= cur_close / prev_close[c]
            prev_close[c] = cur_close

            # ATR止损：简化为固定比例止损（-8%），因为单期内无法计算完整ATR
            loss_pct = cur_close / entry_price - 1
            if loss_pct <= -0.08:
                active_codes.discard(c)
                stopped_codes.add(c)

    # 计算等权组合收益
    valid_navs = [stock_nav[c] - 1 for c in codes if stock_nav.get(c) is not None]
    period_ret = np.mean(valid_navs) if valid_navs else 0.0
    return period_ret, len(valid_navs), len(stopped_codes)


def run_one_backtest(idx, df_all, bt_start, bt_end, train_years, with_trend_position=False):
    """运行单次回测，返回结果字典"""
    train_end = pd.Timestamp(bt_start)
    df_train = df_all[df_all["trade_date"] < train_end].copy()

    # 训练用宏观级别 regime 标签（人工划分，避免趋势模型噪声污染）
    df_train["regime"] = assign_macro_regime(df_train["trade_date"])

    # 推理用趋势模型（仓位控制，灵敏）
    cfg_trend = TrendModelConfig(
        confirmation_periods=3,
        exit_tolerance=5,
        min_hold_periods=7,
    )
    trend = TrendModelRuleV2(cfg_trend)

    cfg = SelectionConfig(model_type="lgbm", label_neutralized=True, ic_filter_enabled=True, industry_col="industry_l1")

    single = StockSelector(copy.deepcopy(cfg))
    single.fit(df_train)

    regime_model = RegimeStockSelector(RegimeSelectionConfig(base_config=copy.deepcopy(cfg)))
    regime_model.fit(df_train, df_train["regime"])

    # 预计算全量特征（修复：避免单日数据导致时序特征全NaN）
    df_prepared = single.prepare_features(df_all)

    idx_bt = idx[idx["date"] <= bt_end].copy()
    # 选股regime：MA均线（稳定）；仓位regime：趋势模型（灵敏）
    d2s_ma = build_ma_regime(idx_bt)
    res_bt = trend.predict(idx_bt, return_history=True)
    d2s_trend = dict(zip(pd.to_datetime(idx_bt["date"]).values, res_bt.diagnostics["states"]))

    bt_dates = sorted(
        df_all[(df_all["trade_date"] >= bt_start) & (df_all["trade_date"] <= bt_end)]["trade_date"].unique()
    )
    rb_dates = bt_dates[::HOLD_DAYS]

    # 增强风控实例
    risk_ctrl = EnhancedRiskControl(RiskControlConfig())

    # 成本模型
    cost_model = create_default_cost_model()
    PORTFOLIO_VALUE = 10_000_000  # 假设1000万组合

    results = {
        "single": [],
        "regime": [],
        "trend_single": [],
        "trend_regime": [],
        "risk_regime": [],
        "risk_trend_regime": [],
    }

    # 风控策略的累计净值（用于计算回撤）
    risk_nav = 1.0
    risk_trend_nav = 1.0
    risk_nav_peak = 1.0
    risk_trend_nav_peak = 1.0
    entry_prices_cache = {}  # 入场价缓存
    risk_stats = {"stop_count": 0, "total_cost": 0.0}

    # 跟踪每个策略的上期持仓（用于计算换手率）
    prev_holdings = {
        "single": set(),
        "regime": set(),
        "trend_single": set(),
        "trend_regime": set(),
        "risk_regime": set(),
        "risk_trend_regime": set(),
    }

    for i, rb in enumerate(rb_dates):
        hold_end = rb_dates[i + 1] if i + 1 < len(rb_dates) else bt_dates[-1]
        df_day = df_prepared[df_prepared["trade_date"] == rb]
        if len(df_day) < 50:
            continue
        # 选股用MA regime（稳定），仓位用趋势模型regime（灵敏）
        regime = d2s_ma.get(pd.Timestamp(rb), 1)
        rname = REGIME_NAMES.get(regime, "?")
        trend_regime = d2s_trend.get(pd.Timestamp(rb), 1)
        pos_ratio = REGIME_POSITION.get(trend_regime, 0.6)

        try:
            pred_s = single.predict_prepared(df_day)
            top_s = pred_s.nlargest(TOP_N, "score")["ts_code"].tolist()
            pred_r = regime_model.predict_prepared(df_day, regime)
            top_r = pred_r.nlargest(TOP_N, "score")["ts_code"].tolist()

            # 获取 Regime 模型的 confidence（Top30 平均 confidence）
            confidence = (
                float(pred_r.nlargest(TOP_N, "score")["confidence"].mean()) if "confidence" in pred_r.columns else 0.7
            )

            hold = df_all[(df_all["trade_date"] > rb) & (df_all["trade_date"] <= hold_end)]

            # ── 原有4种策略（加入成本扣减） ──
            for codes, key in [(top_s, "single"), (top_r, "regime")]:
                rets = []
                for c in codes:
                    s = hold[hold["ts_code"] == c].sort_values("trade_date")
                    if len(s) >= 2:
                        rets.append(s["close"].iloc[-1] / s["close"].iloc[0] - 1)
                period_ret = np.mean(rets) if rets else 0

                # 计算换手成本
                new_set = set(codes)
                old_set = prev_holdings[key]
                turnover = 1.0 - len(new_set & old_set) / max(len(new_set), 1)  # 换手率
                cost = cost_model.calculate_turnover_cost(turnover, PORTFOLIO_VALUE)
                period_ret -= cost
                prev_holdings[key] = new_set

                results[key].append({"date": rb, "regime": rname, "ret": period_ret, "n": len(rets), "cost": cost})
                # 趋势联动版本
                trend_ret = period_ret * pos_ratio + 0 * (1 - pos_ratio)
                results[f"trend_{key}"].append(
                    {
                        "date": rb,
                        "regime": rname,
                        "ret": trend_ret,
                        "n": len(rets),
                        "pos": pos_ratio,
                        "cost": cost * pos_ratio,
                    }
                )
                prev_holdings[f"trend_{key}"] = new_set

            # ── 新增：Regime + 增强风控 ──
            # 计算带止损的期间收益
            raw_ret, n_stocks, n_stopped = _compute_period_return_with_stop_loss(
                df_all, top_r, rb, hold_end, entry_prices_cache, risk_ctrl
            )
            risk_stats["stop_count"] += n_stopped

            # 换手成本（风控策略用 regime 选股，换手率同 regime）
            risk_new_set = set(top_r)
            risk_turnover = 1.0 - len(risk_new_set & prev_holdings["risk_regime"]) / max(len(risk_new_set), 1)
            risk_cost = cost_model.calculate_turnover_cost(risk_turnover, PORTFOLIO_VALUE)
            risk_stats["total_cost"] += risk_cost
            raw_ret -= risk_cost
            prev_holdings["risk_regime"] = risk_new_set

            # 计算当前回撤
            risk_drawdown = risk_nav / risk_nav_peak - 1 if risk_nav_peak > 0 else 0

            # 动态仓位 = confidence仓位 × 回撤降仓
            risk_position = risk_ctrl.compute_dynamic_position(
                confidence=confidence,
                current_drawdown=risk_drawdown,
                daily_return=raw_ret,  # 用期间收益近似
            )
            risk_ret = raw_ret * risk_position
            risk_nav *= 1 + risk_ret
            risk_nav_peak = max(risk_nav_peak, risk_nav)
            results["risk_regime"].append(
                {
                    "date": rb,
                    "regime": rname,
                    "ret": risk_ret,
                    "n": n_stocks,
                    "pos": risk_position,
                    "stopped": n_stopped,
                    "cost": risk_cost,
                }
            )

            # ── 新增：趋势 + Regime + 增强风控 ──
            # 趋势仓位 × 风控仓位（双重保护）
            risk_trend_drawdown = risk_trend_nav / risk_trend_nav_peak - 1 if risk_trend_nav_peak > 0 else 0
            risk_trend_position = risk_ctrl.compute_dynamic_position(
                confidence=confidence,
                current_drawdown=risk_trend_drawdown,
                daily_return=raw_ret,
            )
            # 取趋势仓位和风控仓位的较小值（更保守）
            combined_position = min(pos_ratio, risk_trend_position)
            risk_trend_ret = raw_ret * combined_position
            risk_trend_nav *= 1 + risk_trend_ret
            risk_trend_nav_peak = max(risk_trend_nav_peak, risk_trend_nav)
            prev_holdings["risk_trend_regime"] = risk_new_set
            results["risk_trend_regime"].append(
                {
                    "date": rb,
                    "regime": rname,
                    "ret": risk_trend_ret,
                    "n": n_stocks,
                    "pos": combined_position,
                    "stopped": n_stopped,
                    "cost": risk_cost,
                }
            )

        except Exception:
            pass

    # 沪深300
    idx_period = idx[(idx["date"] >= bt_start) & (idx["date"] <= bt_end)].sort_values("date")
    idx_ret = idx_period["close"].iloc[-1] / idx_period["close"].iloc[0] - 1 if len(idx_period) >= 2 else 0

    return results, idx_ret, risk_stats


def calc_stats(records):
    if not records:
        return {}
    rets = [r["ret"] for r in records]
    nav = [1.0]
    for r in rets:
        nav.append(nav[-1] * (1 + r))
    total = nav[-1] - 1
    avg = np.mean(rets)
    std = np.std(rets)
    sharpe = avg / (std + 1e-8) * np.sqrt(252 / HOLD_DAYS)
    max_dd = min(np.array(nav) / np.maximum.accumulate(nav) - 1)
    win = sum(1 for r in rets if r > 0) / len(rets)
    return {"total": total, "sharpe": sharpe, "max_dd": max_dd, "win_rate": win, "avg": avg, "n_periods": len(rets)}


def main():
    print("加载数据...")
    idx, df_all = load_all_data()

    periods = [
        ("2021-02-01", "2024-02-01", list(range(2020, 2021))),
        ("2024-09-10", "2026-01-01", list(range(2020, 2025))),
    ]

    ALL_STRATEGIES = [
        ("single", "单一模型"),
        ("regime", "Regime模型"),
        ("trend_single", "趋势+单一"),
        ("trend_regime", "趋势+Regime"),
        ("risk_regime", "风控+Regime"),
        ("risk_trend_regime", "风控+趋势+Regime"),
    ]

    all_results = {}
    for bt_start, bt_end, train_years in periods:
        label = f"{bt_start[:7]}~{bt_end[:7]}"
        print(f"\n{'='*60}")
        print(f"回测区间: {label}")
        print(f"训练数据: {train_years}")
        print(f"{'='*60}")

        results, idx_ret, risk_stats = run_one_backtest(idx, df_all, bt_start, bt_end, train_years)

        stats = {}
        for key, name in ALL_STRATEGIES:
            s = calc_stats(results[key])
            stats[key] = s
            if not s:
                continue
            print(f"\n  {name}:")
            print(f"    累计收益: {s['total']:+.2%}")
            print(f"    Sharpe: {s['sharpe']:.2f}")
            print(f"    最大回撤: {s['max_dd']:.2%}")
            print(f"    胜率: {s['win_rate']:.1%} ({s['n_periods']}期)")

        print(f"\n  沪深300: {idx_ret:+.2%}")
        print(f"\n  风控统计: 止损触发 {risk_stats['stop_count']} 次")

        # 分regime
        print("\n  分Regime:")
        for rname in ["bull", "neutral", "bear"]:
            parts = []
            for key, name in ALL_STRATEGIES:
                r_rets = [r["ret"] for r in results[key] if r["regime"] == rname]
                if r_rets:
                    parts.append(f"{name}={np.mean(r_rets):+.2%}")
            if parts:
                n = len([r for r in results["single"] if r["regime"] == rname])
                print(f"    [{rname:7s}] {', '.join(parts)}, 期数={n}")

        all_results[label] = {"stats": stats, "idx_ret": idx_ret, "records": results}

    # 输出汇总
    print(f"\n\n{'='*90}")
    print("全区间汇总对比")
    print(f"{'='*90}")
    print(f"{'区间':<20} {'策略':<18} {'累计收益':>10} {'Sharpe':>8} {'最大回撤':>10} {'胜率':>8}")
    print("-" * 90)
    for label, data in all_results.items():
        for key, name in ALL_STRATEGIES:
            s = data["stats"].get(key)
            if not s:
                continue
            print(
                f"{label:<20} {name:<18} {s['total']:>+10.2%} {s['sharpe']:>8.2f} {s['max_dd']:>10.2%} {s['win_rate']:>8.1%}"
            )
        print(f"{label:<20} {'沪深300':<18} {data['idx_ret']:>+10.2%}")
        print("-" * 90)


if __name__ == "__main__":
    start_time = datetime.now().timestamp()
    failure_reason = None
    try:
        main()
    except Exception as exc:
        failure_reason = str(exc)
        raise
    finally:
        log_task_summary(
            logger,
            task_name="backtest_multi_period",
            window=None,
            elapsed_s=datetime.now().timestamp() - start_time,
            error=failure_reason,
        )
