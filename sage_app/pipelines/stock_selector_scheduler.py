#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
选股任务调度（APScheduler）

能力：
1) 每月首个交易日自动重训（stock_selector_monthly）
2) 每周固定时点自动出信号（stock_selector_weekly_signal）

用法：
  python sage_app/pipelines/stock_selector_scheduler.py --mode once_monthly
  python sage_app/pipelines/stock_selector_scheduler.py --mode once_weekly
  python sage_app/pipelines/stock_selector_scheduler.py --mode cron
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import setup_logger
from scripts.stock.run_stock_selector_monthly import _resolve_tushare_root


def _is_first_trading_day_from_dates(trade_dates: pd.Series, reference_date: str) -> bool:
    if trade_dates.empty:
        return False
    dt = pd.to_datetime(trade_dates.astype(str), errors="coerce").dropna()
    if dt.empty:
        return False
    ref = pd.to_datetime(reference_date)
    month_dates = dt[(dt.dt.year == ref.year) & (dt.dt.month == ref.month)]
    if month_dates.empty:
        return False
    first_trade = month_dates.min().normalize()
    return ref.normalize() == first_trade


def _is_first_trading_day(tushare_root: Path, reference_date: str) -> bool:
    year = int(reference_date[:4])
    path = tushare_root / "daily" / f"daily_{year}.parquet"
    if not path.exists():
        return False
    df = pd.read_parquet(path, columns=["trade_date"])
    return _is_first_trading_day_from_dates(df["trade_date"], reference_date)


def _run_subprocess(command: list[str], logger) -> None:
    logger.info("执行命令: %s", " ".join(command))
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr:
            logger.error(result.stderr.strip())
        raise RuntimeError(f"任务执行失败，退出码={result.returncode}")
    if result.stderr:
        logger.warning(result.stderr.strip())


def _run_monthly_job(args, logger) -> None:
    today = datetime.now().strftime("%Y%m%d")
    tushare_root = _resolve_tushare_root(args.data_dir)
    if not _is_first_trading_day(tushare_root, today):
        logger.info("今日(%s)不是当月首个交易日，跳过月度重训", today)
        return

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/stock/run_stock_selector_monthly.py"),
        "--train-lookback-days", str(args.train_lookback_days),
        "--top-n", str(args.top_n),
    ]
    if args.allow_rule_fallback:
        command.append("--allow-rule-fallback")
    if args.data_dir:
        command.extend(["--data-dir", args.data_dir])
    if args.output_root:
        command.extend(["--output-root", args.output_root])
    _run_subprocess(command, logger)


def _run_weekly_job(args, logger) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/stock/run_stock_selector_weekly_signal.py"),
        "--feature-lookback-days", str(args.feature_lookback_days),
        "--top-n", str(args.top_n),
    ]
    if args.data_dir:
        command.extend(["--data-dir", args.data_dir])
    if args.output_root:
        command.extend(["--output-root", args.output_root])
    _run_subprocess(command, logger)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["once_monthly", "once_weekly", "cron"], default="cron")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--train-lookback-days", type=int, default=900)
    parser.add_argument("--feature-lookback-days", type=int, default=260)
    parser.add_argument("--allow-rule-fallback", action="store_true")

    parser.add_argument("--monthly-hour", type=int, default=20)
    parser.add_argument("--monthly-minute", type=int, default=15)
    parser.add_argument("--weekly-day-of-week", type=str, default="fri")
    parser.add_argument("--weekly-hour", type=int, default=20)
    parser.add_argument("--weekly-minute", type=int, default=30)
    args = parser.parse_args()

    logger = setup_logger("stock_selector_scheduler", module="jobs")

    if args.mode == "once_monthly":
        _run_monthly_job(args, logger)
        return
    if args.mode == "once_weekly":
        _run_weekly_job(args, logger)
        return

    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler()
    scheduler.add_job(
        _run_monthly_job,
        CronTrigger(day="1-7", day_of_week="mon-fri", hour=args.monthly_hour, minute=args.monthly_minute),
        args=[args, logger],
        id="stock_selector_monthly_retrain",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_weekly_job,
        CronTrigger(day_of_week=args.weekly_day_of_week, hour=args.weekly_hour, minute=args.weekly_minute),
        args=[args, logger],
        id="stock_selector_weekly_signal",
        replace_existing=True,
    )
    logger.info(
        "调度启动: monthly=1-7工作日 %02d:%02d, weekly=%s %02d:%02d",
        args.monthly_hour, args.monthly_minute,
        args.weekly_day_of_week, args.weekly_hour, args.weekly_minute,
    )
    scheduler.start()


if __name__ == "__main__":
    main()
