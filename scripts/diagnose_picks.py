"""诊断：每期Top5选股明细（单一 vs Regime）"""
import os, sys, copy, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector
from sage_core.stock_selection.regime_stock_selector import (
    RegimeSelectionConfig, RegimeStockSelector, REGIME_NAMES,
)
from scripts.backtest_multi_period import load_all_data, assign_macro_regime, HOLD_DAYS, MACRO_REGIMES

TOP_N = 30

# 加载股票名称
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "tushare")
name_map = {}
ths_path = os.path.join(DATA_ROOT, "concepts", "ths_member.parquet")
if os.path.exists(ths_path):
    ths = pd.read_parquet(ths_path)
    name_map = dict(zip(ths["con_code"], ths["con_name"]))

def run_diagnose(bt_start, bt_end, train_years):
    print(f"\n{'='*70}")
    print(f"区间: {bt_start} ~ {bt_end}, 训练: {train_years}")
    print(f"{'='*70}")

    idx, df_all = load_all_data()
    train_end = pd.Timestamp(bt_start)
    df_train = df_all[df_all["trade_date"] < train_end].copy()
    df_train["regime"] = assign_macro_regime(df_train["trade_date"])

    cfg = SelectionConfig(model_type="lgbm", label_neutralized=True,
                          ic_filter_enabled=True, industry_col="industry_l1")

    single = StockSelector(copy.deepcopy(cfg))
    single.fit(df_train)
    print(f"单一模型特征: {single.feature_cols}")

    regime_model = RegimeStockSelector(RegimeSelectionConfig(base_config=copy.deepcopy(cfg)))
    regime_model.fit(df_train, df_train["regime"])
    for r, m in regime_model.models.items():
        print(f"Regime[{REGIME_NAMES[r]}]特征: {m.feature_cols}")

    df_prepared = single.prepare_features(df_all)

    idx_bt = idx[idx["date"] <= bt_end].copy()
    from scripts.backtest_multi_period import build_ma_regime
    d2s_bt = build_ma_regime(idx_bt)

    bt_dates = sorted(df_all[(df_all["trade_date"] >= bt_start) & (df_all["trade_date"] <= bt_end)]["trade_date"].unique())
    rb_dates = bt_dates[::HOLD_DAYS]

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
            def get_ret(code):
                s = hold[hold["ts_code"] == code].sort_values("trade_date")
                return s["close"].iloc[-1] / s["close"].iloc[0] - 1 if len(s) >= 2 else 0

            s_rets = [get_ret(c) for c in top_s["ts_code"].head(5)]
            r_rets = [get_ret(c) for c in top_r["ts_code"].head(5)]
            port_s = np.mean([get_ret(c) for c in top_s["ts_code"]])
            port_r = np.mean([get_ret(c) for c in top_r["ts_code"]])

            print(f"\n--- {str(rb)[:10]} [{rname}] 组合收益: 单一={port_s:+.2%} Regime={port_r:+.2%} ---")

            # 获取市值信息
            day_mv = df_day.set_index("ts_code")["total_mv"] if "total_mv" in df_day.columns else pd.Series(dtype=float)

            print(f"  单一Top5:")
            for j, (_, row) in enumerate(top_s.head(5).iterrows()):
                code = row["ts_code"]
                nm = name_map.get(code, "")
                mv = day_mv.get(code, 0) / 10000 if code in day_mv.index else 0
                ind = df_day[df_day["ts_code"] == code]["industry_l1"].values
                ind = ind[0] if len(ind) > 0 and pd.notna(ind[0]) else ""
                print(f"    {j+1}. {code} {nm:<6s} score={row['score']:.4f} ret={s_rets[j]:+.2%} 市值={mv:.0f}亿 {ind}")

            print(f"  Regime[{rname}]Top5:")
            for j, (_, row) in enumerate(top_r.head(5).iterrows()):
                code = row["ts_code"]
                nm = name_map.get(code, "")
                mv = day_mv.get(code, 0) / 10000 if code in day_mv.index else 0
                ind = df_day[df_day["ts_code"] == code]["industry_l1"].values
                ind = ind[0] if len(ind) > 0 and pd.notna(ind[0]) else ""
                print(f"    {j+1}. {code} {nm:<6s} score={row['score']:.4f} ret={r_rets[j]:+.2%} 市值={mv:.0f}亿 {ind}")

            # 重叠度
            overlap = len(set(top_s["ts_code"]) & set(top_r["ts_code"]))
            print(f"  Top30重叠: {overlap}/30")

        except Exception as e:
            print(f"  {str(rb)[:10]} 失败: {e}")

if __name__ == "__main__":
    print("加载数据...")
    # 区间一（熊市）
    run_diagnose("2021-02-01", "2024-02-01", [2020])
    # 区间二（牛市）
    run_diagnose("2024-09-10", "2026-01-01", list(range(2020, 2025)))
