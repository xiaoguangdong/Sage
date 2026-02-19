#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
趋势主信号定时导出（APScheduler）

用法:
  python sage_app/pipelines/trend_signal_scheduler.py --mode once
  python sage_app/pipelines/trend_signal_scheduler.py --mode cron --hour 18 --minute 0
  python sage_app/pipelines/trend_signal_scheduler.py --mode interval --interval-minutes 120
"""

import argparse
import sys
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_tushare_root, setup_logger
from scripts.models.export_trend_signal import export_trend_main_signal


def run_job(timeframe: str, output_dir: Path, data_dir: str | None):
    logger = setup_logger("trend_signal_export", module="jobs")
    logger.info("任务开始: 导出趋势主信号")
    output_file = export_trend_main_signal(timeframe=timeframe, output_dir=output_dir, data_dir=data_dir)
    logger.info(f"任务完成: output={output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["once", "cron", "interval"], default="cron")
    parser.add_argument("--timeframe", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--output-dir", default="data/signals")
    parser.add_argument("--data-dir", default=None, help="Tushare数据根目录")
    parser.add_argument("--hour", type=int, default=18)
    parser.add_argument("--minute", type=int, default=0)
    parser.add_argument("--day-of-week", default="mon-fri")
    parser.add_argument("--interval-minutes", type=int, default=120)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = args.data_dir or str(get_tushare_root())

    if args.mode == "once":
        run_job(args.timeframe, output_dir, data_dir)
        return

    scheduler = BlockingScheduler()

    if args.mode == "interval":
        scheduler.add_job(
            run_job,
            "interval",
            minutes=args.interval_minutes,
            args=[args.timeframe, output_dir, data_dir],
            id="trend_signal_interval",
            replace_existing=True,
        )
    else:
        scheduler.add_job(
            run_job,
            CronTrigger(day_of_week=args.day_of_week, hour=args.hour, minute=args.minute),
            args=[args.timeframe, output_dir, data_dir],
            id="trend_signal_cron",
            replace_existing=True,
        )

    scheduler.start()


if __name__ == "__main__":
    main()
