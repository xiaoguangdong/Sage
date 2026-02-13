#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
导出宏观三大信号（PMI拐点 + 利差阈值 + 行业景气度 + 北向行业配置）

用法:
  python scripts/models/macro/export_macro_signals.py
  python scripts/models/macro/export_macro_signals.py --output-dir data/signals --delay-days 2
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from sage_core.models.signal_indicators import MacroSignal, IndustryProsperity, NorthboundFlow


def apply_delay(df: pd.DataFrame, date_col: str, delay_days: int) -> pd.DataFrame:
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col]) + pd.Timedelta(days=delay_days)
    return result


def export_macro_signals(output_dir: Path, delay_days: int, spread_threshold: float, spread_mode: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 宏观信号
    pmi_out = None
    spread_out = None
    try:
        macro = MacroSignal()
        macro_result = macro.get_macro_signal(spread_threshold=spread_threshold, spread_mode=spread_mode)

        pmi_sig = macro_result['pmi_signal'].copy()
        pmi_sig['date'] = pd.to_datetime(pmi_sig['month'], format='%Y%m')
        pmi_sig = apply_delay(pmi_sig, 'date', delay_days)
        pmi_out = output_dir / f"macro_pmi_signal_{pmi_sig['date'].iloc[-1].strftime('%Y%m%d')}.parquet"
        pmi_sig[['date', 'pmi', 'pmi_diff', 'turning_point']].to_parquet(pmi_out, index=False)

        spread_sig = macro_result['spread_signal'].copy()
        spread_sig = apply_delay(spread_sig, 'trade_date', delay_days)
        spread_out = output_dir / f"macro_yield_spread_{spread_sig['trade_date'].iloc[-1].strftime('%Y%m%d')}.parquet"
        spread_sig[['trade_date', 'yield_10y', 'yield_2y', 'spread', 'spread_signal']].to_parquet(spread_out, index=False)
    except FileNotFoundError:
        pass

    # 2) 行业景气度（个股层）
    prosperity_out = None
    try:
        prosperity = IndustryProsperity()
        prosperity_df = prosperity.get_prosperity_signal()
        if prosperity_df is not None and len(prosperity_df) > 0:
            prosperity_df['end_date'] = pd.to_datetime(prosperity_df['end_date'])
            prosperity_df = apply_delay(prosperity_df, 'end_date', delay_days)
            prosperity_out = output_dir / f"industry_prosperity_{prosperity_df['end_date'].iloc[-1].strftime('%Y%m%d')}.parquet"
            prosperity_df[['ts_code', 'end_date', 'tr_yoy', 'gross_margin', 'revenue_acceleration', 'prosperity_signal']].to_parquet(
                prosperity_out, index=False
            )
    except FileNotFoundError:
        pass

    # 3) 北向资金信号（市场级 + 行业配置比例）
    flow_out = None
    industry_out = None
    try:
        flow = NorthboundFlow()
        flow_df = flow.get_flow_signal()
        flow_df = apply_delay(flow_df, 'trade_date', delay_days)
        flow_out = output_dir / f"northbound_flow_{flow_df['trade_date'].iloc[-1].strftime('%Y%m%d')}.parquet"
        flow_df[['trade_date', 'north_money', 'flow_ma', 'upper_band', 'flow_signal']].to_parquet(flow_out, index=False)

        industry_flow_df = flow.get_industry_flow_signal()
        if industry_flow_df is not None and len(industry_flow_df) > 0:
            industry_flow_df = apply_delay(industry_flow_df, 'trade_date', delay_days)
            industry_out = output_dir / f"northbound_industry_ratio_{industry_flow_df['trade_date'].iloc[-1].strftime('%Y%m%d')}.parquet"
            industry_flow_df[
                ['industry_code', 'industry_name', 'trade_date', 'industry_ratio', 'ratio_signal']
            ].to_parquet(industry_out, index=False)
    except FileNotFoundError:
        pass

    return {
        "pmi": pmi_out,
        "spread": spread_out,
        "prosperity": prosperity_out,
        "northbound_flow": flow_out,
        "northbound_industry": industry_out
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/signals")
    parser.add_argument("--delay-days", type=int, default=2)
    parser.add_argument("--spread-threshold", type=float, default=0.5)
    parser.add_argument("--spread-mode", type=str, default="threshold", choices=["threshold", "bollinger"])
    args = parser.parse_args()

    outputs = export_macro_signals(
        output_dir=Path(args.output_dir),
        delay_days=args.delay_days,
        spread_threshold=args.spread_threshold,
        spread_mode=args.spread_mode
    )

    print("宏观信号导出完成:")
    for name, path in outputs.items():
        if path is None:
            print(f"  - {name}: 无数据")
        else:
            print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
