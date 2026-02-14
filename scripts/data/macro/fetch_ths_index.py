#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
获取同花顺板块指数数据（ths_index / ths_daily）

接口说明：
- ths_index：获取同花顺板块指数（单次可全量）
- ths_daily：获取板块指数行情（按指数代码/日期循环）
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import tushare as ts

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import disable_proxy
from scripts.data.macro.paths import CONCEPTS_DIR
from scripts.data.macro.tushare_auth import get_tushare_token


def get_with_retry(pro, api_name, params, max_retries=3, sleep_time=60):
    for attempt in range(max_retries):
        try:
            api_func = getattr(pro, api_name)
            df = api_func(**params)
            time.sleep(sleep_time)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30
                print(f"  请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"  请求失败，已达到最大重试次数: {e}")
                return None


def fetch_ths_index(pro, output_dir: Path):
    print("\n" + "=" * 80)
    print("获取同花顺板块指数列表 (ths_index)")
    print("=" * 80)

    output_file = output_dir / "ths_index.parquet"
    df = get_with_retry(pro, "ths_index", {}, max_retries=3, sleep_time=1)
    if df is None or df.empty:
        print("  获取失败或无数据")
        return
    df.to_parquet(output_file, index=False)
    print(f"  已保存到 {output_file}，行数 {len(df)}")


def fetch_ths_daily(pro, output_dir: Path, start_date: str, end_date: str, sleep_seconds: int = 40):
    print("\n" + "=" * 80)
    print("获取同花顺板块指数行情 (ths_daily)")
    print("=" * 80)

    index_file = output_dir / "ths_index.parquet"
    if not index_file.exists():
        print("  缺少 ths_index，请先执行 ths_index 拉取")
        return

    output_file = output_dir / "ths_daily.parquet"
    progress_file = output_dir / "ths_daily_progress.txt"

    last_code = ""
    last_month = ""
    if progress_file.exists():
        content = progress_file.read_text(encoding="utf-8").strip()
        if content:
            parts = content.split(",")
            last_code = parts[0]
            last_month = parts[1] if len(parts) > 1 else ""
        print(f"  找到进度文件，上次到: {last_code} {last_month}")

    index_df = pd.read_parquet(index_file)
    if "ts_code" not in index_df.columns:
        print("  ths_index 缺少 ts_code 字段")
        return

    codes = index_df["ts_code"].dropna().astype(str).unique().tolist()
    if last_code and last_code in codes:
        start_idx = codes.index(last_code) + 1
    else:
        start_idx = 0

    existing = pd.read_parquet(output_file) if output_file.exists() else pd.DataFrame()

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    months = []
    current_dt = datetime(start_dt.year, start_dt.month, 1)
    while current_dt <= end_dt:
        months.append(current_dt)
        if current_dt.month == 12:
            current_dt = datetime(current_dt.year + 1, 1, 1)
        else:
            current_dt = datetime(current_dt.year, current_dt.month + 1, 1)

    for i, code in enumerate(codes[start_idx:], start=start_idx):
        print(f"[{i+1}/{len(codes)}] 获取 {code} 行情...")
        for month_start in months:
            month_end = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            if month_end > end_dt:
                month_end = end_dt
            start_str = month_start.strftime("%Y%m%d")
            end_str = month_end.strftime("%Y%m%d")

            if code == last_code and last_month and end_str <= last_month:
                continue

            df = get_with_retry(
                pro,
                "ths_daily",
                {"ts_code": code, "start_date": start_str, "end_date": end_str},
                max_retries=3,
                sleep_time=sleep_seconds,
            )
            if df is not None and not df.empty:
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["ts_code", "trade_date"])
                combined.to_parquet(output_file, index=False)
                existing = combined
                print(f"  {start_str}-{end_str} 已累计保存 {len(existing)} 条")
            else:
                print(f"  {start_str}-{end_str} 无数据或获取失败")

            progress_file.write_text(f"{code},{end_str}", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default="20200101")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--sleep-seconds", type=int, default=40)
    parser.add_argument("--only-index", action="store_true")
    parser.add_argument("--only-daily", action="store_true")
    args = parser.parse_args()

    disable_proxy()
    token = get_tushare_token()
    pro = ts.pro_api(token)

    output_dir = Path(CONCEPTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.only_daily:
        fetch_ths_index(pro, output_dir)
    if not args.only_index:
        fetch_ths_daily(pro, output_dir, args.start_date, args.end_date, sleep_seconds=args.sleep_seconds)


if __name__ == "__main__":
    main()
