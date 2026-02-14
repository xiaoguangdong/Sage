#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
获取同花顺板块指数数据（ths_index / ths_daily）
薄封装：逻辑统一放在 tushare_tasks
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.tushare_tasks import run_ths_index, run_ths_daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default="20200101")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--sleep-seconds", type=int, default=40)
    parser.add_argument("--only-index", action="store_true")
    parser.add_argument("--only-daily", action="store_true")
    parser.add_argument("--all-by-month", action="store_true", help="按月全量分页拉取（不输入指数）")
    args = parser.parse_args()

    if not args.only_daily:
        run_ths_index()
    if not args.only_index:
        run_ths_daily(
            start_date=args.start_date,
            end_date=args.end_date,
            sleep_seconds=args.sleep_seconds,
            all_by_month=args.all_by_month,
        )


if __name__ == "__main__":
    main()
