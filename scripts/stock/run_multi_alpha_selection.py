#!/usr/bin/env python3
"""
测试脚本：多逻辑选股模型 (Value / Growth / Frontier)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import logging

import pandas as pd

# 确保可以从项目根目录导入
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sage_core.stock_selection.multi_alpha_selector import MultiAlphaStockSelector
from scripts.data._shared.runtime import get_tushare_root


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("multi_alpha_selection")


def _find_latest_trade_date(data_dir: Path) -> str:
    daily_dir = data_dir / "daily"
    years = sorted([p for p in daily_dir.glob("daily_*.parquet")])
    if not years:
        raise FileNotFoundError(f"未找到日线数据目录: {daily_dir}")

    latest_date = None
    for path in years[::-1]:
        df = pd.read_parquet(path, columns=["trade_date"])
        max_date = df["trade_date"].max()
        if max_date and (latest_date is None or max_date > latest_date):
            latest_date = max_date
        if latest_date:
            break

    if latest_date is None:
        raise ValueError("无法在日线数据中找到有效 trade_date")

    return str(latest_date)


def main():
    parser = argparse.ArgumentParser(description="多逻辑选股模型测试脚本")
    parser.add_argument("--date", type=str, default=None, help="交易日期 YYYYMMDD (默认自动取最新)")
    parser.add_argument("--top-n", type=int, default=30, help="每个子组合选股数量")
    parser.add_argument("--allocation", type=str, default="fixed", choices=["fixed", "regime"], help="组合权重方式")
    parser.add_argument("--regime", type=str, default="sideways", choices=["bear", "sideways", "bull"], help="市场状态")
    parser.add_argument("--data-dir", type=str, default=str(get_tushare_root()), help="数据目录")
    parser.add_argument("--output", type=str, default=None, help="输出CSV路径")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    trade_date = args.date or _find_latest_trade_date(data_dir)

    logger.info("运行多逻辑选股: trade_date=%s", trade_date)

    selector = MultiAlphaStockSelector(data_dir=str(data_dir))
    results = selector.select(
        trade_date=trade_date,
        top_n=args.top_n,
        allocation_method=args.allocation,
        regime=args.regime,
    )

    for name in ["value", "growth", "frontier", "combined"]:
        df = results[name]
        logger.info("\n==== %s Top %d ====" % (name.upper(), args.top_n))
        logger.info("\n%s", df.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results["all_scores"].to_csv(output_path, index=False)
        logger.info("结果已保存: %s", output_path)


if __name__ == "__main__":
    main()
