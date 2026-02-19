#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str]) -> None:
    print("执行:", " ".join(command))
    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="ths_daily 按月全量下载 + 完整性校验")
    parser.add_argument("--start-date", type=str, default="20200101", help="下载起始日期 YYYYMMDD")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y%m%d"), help="下载结束日期 YYYYMMDD")
    parser.add_argument("--sleep-seconds", type=int, default=40, help="接口调用间隔秒数")
    parser.add_argument("--output-root", type=str, default=None, help="覆盖下载输出根目录")
    parser.add_argument("--no-resume", action="store_true", help="关闭断点续传（默认开启）")
    parser.add_argument("--download-only", action="store_true", help="仅下载，不做完整性校验")
    parser.add_argument("--check-only", action="store_true", help="仅做完整性校验，不下载")
    parser.add_argument("--dry-run-download", action="store_true", help="仅预演下载窗口，不发起接口请求")
    parser.add_argument("--max-stale-days", type=int, default=7, help="完整性校验：允许最大滞后天数")
    parser.add_argument("--min-monthly-ratio", type=float, default=0.8, help="完整性校验：月数据量下限比例")
    parser.add_argument("--report-output", type=str, default=None, help="完整性报告输出路径")
    args = parser.parse_args()

    if args.check_only and args.download_only:
        raise SystemExit("--check-only 与 --download-only 不能同时使用")

    if not args.check_only:
        download_cmd = [
            sys.executable,
            "-m",
            "scripts.data.tushare_downloader",
            "--task",
            "ths_daily",
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--sleep-seconds",
            str(args.sleep_seconds),
        ]
        if not args.no_resume:
            download_cmd.append("--resume")
        if args.output_root:
            download_cmd.extend(["--output-root", args.output_root])
        if args.dry_run_download:
            download_cmd.append("--dry-run")
        _run(download_cmd)

    if not args.download_only:
        check_cmd = [
            sys.executable,
            "-m",
            "scripts.monitoring.check_ths_daily_completeness",
            "--max-stale-days",
            str(args.max_stale_days),
            "--min-monthly-ratio",
            str(args.min_monthly_ratio),
        ]
        if args.report_output:
            check_cmd.extend(["--output", args.report_output])
        _run(check_cmd)


if __name__ == "__main__":
    main()
