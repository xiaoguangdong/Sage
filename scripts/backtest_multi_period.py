"""选股模型多区间回测 — 支持趋势模型仓位联动"""
import os, sys, copy, logging, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector
from sage_core.stock_selection.regime_stock_selector import (
    RegimeSelectionConfig, RegimeStockSelector, REGIME_NAMES,
)
from sage_core.trend.trend_model import TrendModelConfig, TrendModelRuleV2

logging.basicConfig(level=logging.WARNING)
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "tushare")
HOLD_DAYS, TOP_N = 20, 30

# 趋势联动仓位
REGIME_POSITION = {0: 0.3, 1: 0.6, 2: 1.0}  # bear/neutral/bull


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
    cols = [c for c in ["ts_code", "trade_date", "pe_ttm", "pb", "turnover_rate", "total_mv", "circ_mv"] if c in basic.columns]
    df = stk.merge(basic[cols], on=["ts_code", "trade_date"], how="left")

    member_path = os.path.join(DATA_ROOT, "sw_industry", "sw_index_member.parquet")
    ind_path = os.path.join(DATA_ROOT, "sw_industry", "sw_industry_l1.parquet")
    if os.path.exists(member_path) and os.path.exists(ind_path):
        member = pd.read_parquet(member_path)
        ind = pd.read_parquet(ind_path)
        # 只取当前有效成员
        current = member[member["is_new"] == "Y"][["index_code", "con_code"]].drop_duplicates("con_code")
        current = current.merge(ind[["index_code", "industry_name"]], on="index_code", how="left")
        df = df.merge(current[["con_code", "industry_name"]].rename(columns={"con_code": "ts_code", "industry_name": "industry_l1"}),
                      on="ts_code", how="left")

    return idx, df


def run_one_backtest(idx, df_all, bt_start, bt_end, train_years, with_trend_position=False):
    """运行单次回测，返回结果字典"""
    train_end = pd.Timestamp(bt_start)
    df_train = df_all[df_all["trade_date"] < train_end].copy()

    # 使用与test_trend_model_v2.py一致的配置（更敏感，识别及时）
    cfg_trend = TrendModelConfig(
        confirmation_periods=3,
        exit_tolerance=5,
        min_hold_periods=7,
    )
    trend = TrendModelRuleV2(cfg_trend)
    idx_train = idx[idx["date"] < train_end].copy()
    res = trend.predict(idx_train, return_history=True)
    d2s = dict(zip(pd.to_datetime(idx_train["date"]).values, res.diagnostics["states"]))
    df_train["regime"] = df_train["trade_date"].map(lambda d: d2s.get(pd.Timestamp(d), 1))

    cfg = SelectionConfig(model_type="lgbm", label_neutralized=True,
                          ic_filter_enabled=True, ic_threshold=0.02,
                          ic_ir_threshold=0.3, industry_col="industry_l1")

    single = StockSelector(copy.deepcopy(cfg))
    single.fit(df_train)

    regime_model = RegimeStockSelector(RegimeSelectionConfig(base_config=copy.deepcopy(cfg)))
    regime_model.fit(df_train, df_train["regime"])

    idx_bt = idx[idx["date"] <= bt_end].copy()
    res_bt = trend.predict(idx_bt, return_history=True)
    d2s_bt = dict(zip(pd.to_datetime(idx_bt["date"]).values, res_bt.diagnostics["states"]))

    bt_dates = sorted(df_all[(df_all["trade_date"] >= bt_start) & (df_all["trade_date"] <= bt_end)]["trade_date"].unique())
    rb_dates = bt_dates[::HOLD_DAYS]

    results = {"single": [], "regime": [], "trend_single": [], "trend_regime": []}

    for i, rb in enumerate(rb_dates):
        hold_end = rb_dates[i + 1] if i + 1 < len(rb_dates) else bt_dates[-1]
        df_day = df_all[df_all["trade_date"] == rb]
        if len(df_day) < 50:
            continue
        regime = d2s_bt.get(pd.Timestamp(rb), 1)
        rname = REGIME_NAMES.get(regime, "?")
        pos_ratio = REGIME_POSITION.get(regime, 0.6)

        try:
            pred_s = single.predict(df_day)
            top_s = pred_s.nlargest(TOP_N, "score")["ts_code"].tolist()
            pred_r = regime_model.predict(df_day, regime)
            top_r = pred_r.nlargest(TOP_N, "score")["ts_code"].tolist()

            hold = df_all[(df_all["trade_date"] > rb) & (df_all["trade_date"] <= hold_end)]

            for codes, key in [(top_s, "single"), (top_r, "regime")]:
                rets = []
                for c in codes:
                    s = hold[hold["ts_code"] == c].sort_values("trade_date")
                    if len(s) >= 2:
                        rets.append(s["close"].iloc[-1] / s["close"].iloc[0] - 1)
                period_ret = np.mean(rets) if rets else 0
                results[key].append({"date": rb, "regime": rname, "ret": period_ret, "n": len(rets)})
                # 趋势联动版本
                trend_ret = period_ret * pos_ratio + 0 * (1 - pos_ratio)  # 空仓部分收益为0
                results[f"trend_{key}"].append({"date": rb, "regime": rname, "ret": trend_ret, "n": len(rets), "pos": pos_ratio})

        except Exception:
            pass

    # 沪深300
    idx_period = idx[(idx["date"] >= bt_start) & (idx["date"] <= bt_end)].sort_values("date")
    idx_ret = idx_period["close"].iloc[-1] / idx_period["close"].iloc[0] - 1 if len(idx_period) >= 2 else 0

    return results, idx_ret


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

    all_results = {}
    for bt_start, bt_end, train_years in periods:
        label = f"{bt_start[:7]}~{bt_end[:7]}"
        print(f"\n{'='*60}")
        print(f"回测区间: {label}")
        print(f"训练数据: {train_years}")
        print(f"{'='*60}")

        results, idx_ret = run_one_backtest(idx, df_all, bt_start, bt_end, train_years)

        stats = {}
        for key in ["single", "regime", "trend_single", "trend_regime"]:
            s = calc_stats(results[key])
            stats[key] = s
            name = {"single": "单一模型", "regime": "Regime模型",
                    "trend_single": "趋势+单一", "trend_regime": "趋势+Regime"}[key]
            print(f"\n  {name}:")
            print(f"    累计收益: {s['total']:+.2%}")
            print(f"    Sharpe: {s['sharpe']:.2f}")
            print(f"    最大回撤: {s['max_dd']:.2%}")
            print(f"    胜率: {s['win_rate']:.1%} ({s['n_periods']}期)")

        print(f"\n  沪深300: {idx_ret:+.2%}")

        # 分regime
        print(f"\n  分Regime:")
        for rname in ["bull", "neutral", "bear"]:
            s_rets = [r["ret"] for r in results["single"] if r["regime"] == rname]
            r_rets = [r["ret"] for r in results["regime"] if r["regime"] == rname]
            ts_rets = [r["ret"] for r in results["trend_single"] if r["regime"] == rname]
            tr_rets = [r["ret"] for r in results["trend_regime"] if r["regime"] == rname]
            if s_rets:
                print(f"    [{rname:7s}] 单一={np.mean(s_rets):+.2%}, Regime={np.mean(r_rets):+.2%}, "
                      f"趋势+单一={np.mean(ts_rets):+.2%}, 趋势+Regime={np.mean(tr_rets):+.2%}, 期数={len(s_rets)}")

        all_results[label] = {"stats": stats, "idx_ret": idx_ret, "records": results}

    # 输出汇总
    print(f"\n\n{'='*70}")
    print("全区间汇总对比")
    print(f"{'='*70}")
    print(f"{'区间':<20} {'策略':<14} {'累计收益':>10} {'Sharpe':>8} {'最大回撤':>10} {'胜率':>8}")
    print("-" * 70)
    for label, data in all_results.items():
        for key, name in [("single", "单一模型"), ("regime", "Regime模型"),
                          ("trend_single", "趋势+单一"), ("trend_regime", "趋势+Regime")]:
            s = data["stats"][key]
            print(f"{label:<20} {name:<14} {s['total']:>+10.2%} {s['sharpe']:>8.2f} {s['max_dd']:>10.2%} {s['win_rate']:>8.1%}")
        print(f"{label:<20} {'沪深300':<14} {data['idx_ret']:>+10.2%}")
        print("-" * 70)


if __name__ == "__main__":
    main()
