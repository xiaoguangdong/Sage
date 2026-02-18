"""展示回测期间每期选股结果（单一模型 vs Regime模型）"""
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

BT_START, BT_END = "2024-09-10", "2026-01-01"
TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024]
HOLD_DAYS, TOP_N = 20, 30
SHOW_TOP = 5


def load_data():
    idx = pd.read_parquet(os.path.join(DATA_ROOT, "index", "index_000300_SH_ohlc.parquet"))
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").reset_index(drop=True)

    frames = []
    for y in TRAIN_YEARS + [2025, 2026]:
        p = os.path.join(DATA_ROOT, "daily", f"daily_{y}.parquet")
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    stk = pd.concat(frames, ignore_index=True)
    stk["trade_date"] = pd.to_datetime(stk["trade_date"], format="%Y%m%d")
    if "turnover" not in stk.columns and "vol" in stk.columns:
        stk["turnover"] = stk["vol"]

    basic = pd.read_parquet(os.path.join(DATA_ROOT, "daily_basic_all.parquet"))
    basic["trade_date"] = pd.to_datetime(basic["trade_date"], format="%Y%m%d")

    # merge
    cols = [c for c in ["ts_code", "trade_date", "pe_ttm", "pb", "turnover_rate", "total_mv", "circ_mv"] if c in basic.columns]
    df = stk.merge(basic[cols], on=["ts_code", "trade_date"], how="left")

    member_path = os.path.join(DATA_ROOT, "sw_industry", "sw_index_member.parquet")
    ind_path = os.path.join(DATA_ROOT, "sw_industry", "sw_industry_l1.parquet")
    if os.path.exists(member_path) and os.path.exists(ind_path):
        member = pd.read_parquet(member_path)
        ind = pd.read_parquet(ind_path)
        current = member[member["is_new"] == "Y"][["index_code", "con_code"]].drop_duplicates("con_code")
        current = current.merge(ind[["index_code", "industry_name"]], on="index_code", how="left")
        df = df.merge(current[["con_code", "industry_name"]].rename(columns={"con_code": "ts_code", "industry_name": "industry_l1"}),
                      on="ts_code", how="left")

    # 股票名称（从同花顺概念成员表获取）
    name_path = os.path.join(DATA_ROOT, "concepts", "ths_member.parquet")
    if os.path.exists(name_path):
        names = pd.read_parquet(name_path)[["con_code", "con_name"]].drop_duplicates("con_code")
        df = df.merge(names.rename(columns={"con_code": "ts_code", "con_name": "name"}), on="ts_code", how="left")

    return idx, df


def main():
    print("加载数据...")
    idx, df_all = load_data()
    has_name = "name" in df_all.columns

    train_end = pd.Timestamp(BT_START)
    df_train = df_all[df_all["trade_date"] < train_end].copy()

    # regime labels（使用与test_trend_model_v2.py一致的配置）
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

    # train
    cfg = SelectionConfig(model_type="lgbm", label_neutralized=True,
                          ic_filter_enabled=True, ic_threshold=0.02,
                          ic_ir_threshold=0.3, industry_col="industry_l1")
    print("训练单一模型...")
    single = StockSelector(copy.deepcopy(cfg))
    single.fit(df_train)

    print("训练Regime模型...")
    regime_model = RegimeStockSelector(RegimeSelectionConfig(base_config=copy.deepcopy(cfg)))
    regime_model.fit(df_train, df_train["regime"])

    # 预计算全量特征（修复：避免单日数据导致时序特征全NaN）
    print("预计算全量特征...")
    df_prepared = single.prepare_features(df_all)

    # bt regime
    idx_bt = idx[idx["date"] <= BT_END].copy()
    res_bt = trend.predict(idx_bt, return_history=True)
    d2s_bt = dict(zip(pd.to_datetime(idx_bt["date"]).values, res_bt.diagnostics["states"]))

    bt_dates = sorted(df_all[(df_all["trade_date"] >= BT_START) & (df_all["trade_date"] <= BT_END)]["trade_date"].unique())
    rb_dates = bt_dates[::HOLD_DAYS]

    all_picks = []

    for i, rb in enumerate(rb_dates):
        hold_end = rb_dates[i + 1] if i + 1 < len(rb_dates) else bt_dates[-1]
        df_day = df_prepared[df_prepared["trade_date"] == rb]
        if len(df_day) < 50:
            continue
        regime = d2s_bt.get(pd.Timestamp(rb), 1)
        rname = REGIME_NAMES.get(regime, "?")

        try:
            pred_s = single.predict_prepared(df_day)
            top_s = pred_s.nlargest(TOP_N, "score")

            pred_r = regime_model.predict_prepared(df_day, regime)
            top_r = pred_r.nlargest(TOP_N, "score")

            # 计算持仓期收益
            hold = df_all[(df_all["trade_date"] > rb) & (df_all["trade_date"] <= hold_end)]
            def calc_ret(codes):
                rets = {}
                for c in codes:
                    s = hold[hold["ts_code"] == c].sort_values("trade_date")
                    if len(s) >= 2:
                        rets[c] = s["close"].iloc[-1] / s["close"].iloc[0] - 1
                return rets

            rets_s = calc_ret(top_s["ts_code"].tolist())
            rets_r = calc_ret(top_r["ts_code"].tolist())

            # 记录
            for code, row in top_s.set_index("ts_code").iterrows():
                name = df_day[df_day["ts_code"] == code]["name"].iloc[0] if has_name and code in df_day["ts_code"].values else ""
                ind = df_day[df_day["ts_code"] == code]["industry_l1"].iloc[0] if "industry_l1" in df_day.columns and code in df_day["ts_code"].values else ""
                all_picks.append({
                    "调仓日": str(rb)[:10], "模型": "单一", "regime": rname,
                    "ts_code": code, "名称": name, "行业": ind,
                    "score": round(row["score"], 4),
                    "持仓收益": round(rets_s.get(code, 0), 4),
                })
            for code, row in top_r.set_index("ts_code").iterrows():
                name = df_day[df_day["ts_code"] == code]["name"].iloc[0] if has_name and code in df_day["ts_code"].values else ""
                ind = df_day[df_day["ts_code"] == code]["industry_l1"].iloc[0] if "industry_l1" in df_day.columns and code in df_day["ts_code"].values else ""
                all_picks.append({
                    "调仓日": str(rb)[:10], "模型": "Regime", "regime": rname,
                    "ts_code": code, "名称": name, "行业": ind,
                    "score": round(row["score"], 4),
                    "持仓收益": round(rets_r.get(code, 0), 4),
                })

            # 打印摘要
            overlap = set(top_s["ts_code"]) & set(top_r["ts_code"])
            avg_s = np.mean(list(rets_s.values())) if rets_s else 0
            avg_r = np.mean(list(rets_r.values())) if rets_r else 0
            print(f"\n{'='*70}")
            print(f"调仓日: {str(rb)[:10]}  [{rname}]  重叠: {len(overlap)}/{TOP_N}")
            print(f"  单一模型 avg={avg_s:+.2%}  |  Regime模型 avg={avg_r:+.2%}")

            # Top N
            print(f"  --- 单一模型 Top{SHOW_TOP} ---")
            for _, r in top_s.head(SHOW_TOP).iterrows():
                c = r["ts_code"]
                nm = str(df_day[df_day["ts_code"]==c]["name"].iloc[0]) if has_name and c in df_day["ts_code"].values else ""
                if nm == "nan": nm = ""
                ret = rets_s.get(c, 0)
                print(f"    {c} {nm:6s} score={r['score']:.4f} ret={ret:+.2%}")

            print(f"  --- Regime模型 Top{SHOW_TOP} ---")
            for _, r in top_r.head(SHOW_TOP).iterrows():
                c = r["ts_code"]
                nm = str(df_day[df_day["ts_code"]==c]["name"].iloc[0]) if has_name and c in df_day["ts_code"].values else ""
                if nm == "nan": nm = ""
                ret = rets_r.get(c, 0)
                print(f"    {c} {nm:6s} score={r['score']:.4f} ret={ret:+.2%}")

        except Exception as e:
            print(f"  {str(rb)[:10]} 失败: {e}")

    # 保存CSV
    df_picks = pd.DataFrame(all_picks)
    out = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_picks.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df_picks.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n选股明细已保存: {out}")
    print(f"总记录: {len(df_picks)}, 调仓期数: {df_picks['调仓日'].nunique()}")

    # 汇总统计
    print(f"\n{'='*70}")
    print("行业分布统计（全部调仓期合计）")
    for model in ["单一", "Regime"]:
        sub = df_picks[df_picks["模型"] == model]
        ind_counts = sub["行业"].value_counts().head(10)
        print(f"\n  [{model}模型] Top10行业:")
        for ind, cnt in ind_counts.items():
            avg = sub[sub["行业"] == ind]["持仓收益"].mean()
            print(f"    {ind:8s}: {cnt:3d}次  avg_ret={avg:+.2%}")


if __name__ == "__main__":
    main()
