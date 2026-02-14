#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
获取东方财富概念指数历史数据 (dc_index)
"""

import pandas as pd
import tushare as ts
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.macro.tushare_auth import get_tushare_token
from scripts.data._shared.runtime import disable_proxy
from scripts.data.macro.paths import CONCEPTS_DIR


def get_with_retry(pro, api_name, params, max_retries=3, sleep_time=40):
    """带重试机制的API请求"""
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


def fetch_concept_index(pro, output_dir, start_date='20200101', end_date=None):
    """获取概念指数历史数据（按月分批）"""
    print("\n" + "=" * 80)
    print("获取概念指数历史数据 (dc_index)")
    print("=" * 80)

    output_file = output_dir / 'dc_index.parquet'
    progress_file = output_dir / 'dc_index_progress.txt'

    # 加载进度
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            last_date = f.read().strip()
        print(f"找到进度文件，上次获取到: {last_date}")
        start_date = str(int(last_date) + 1)
    else:
        print(f"无进度文件，从 {start_date} 开始")

    # 加载已有数据
    if output_file.exists():
        existing_data = pd.read_parquet(output_file)
        print(f"已有数据: {len(existing_data)} 条记录")
        if len(existing_data) > 0:
            existing_dates = set(existing_data['trade_date'].unique())
        else:
            existing_dates = set()
    else:
        existing_data = pd.DataFrame()
        existing_dates = set()

    # 生成日期列表
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.now() - timedelta(days=1) if end_date is None else datetime.strptime(end_date, '%Y%m%d')
    months = []
    current_dt = datetime(start_dt.year, start_dt.month, 1)
    while current_dt <= end_dt:
        months.append(current_dt)
        if current_dt.month == 12:
            current_dt = datetime(current_dt.year + 1, 1, 1)
        else:
            current_dt = datetime(current_dt.year, current_dt.month + 1, 1)

    print(f"需要获取 {len(months)} 个月的概念指数数据")

    all_index = []
    total_months = len(months)

    for i, month_start in enumerate(months):
        month_end = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        if month_end > end_dt:
            month_end = end_dt
        start_str = month_start.strftime('%Y%m%d')
        end_str = month_end.strftime('%Y%m%d')

        print(f"\n[{i+1}/{total_months}] 获取 {start_str} ~ {end_str} 的概念指数数据...")

        offset = 0
        page = 1
        page_data = []

        while True:
            print(f"  第{page}页获取 (offset={offset})...", end=' ')
            df = get_with_retry(
                pro, 'dc_index',
                {'start_date': start_str, 'end_date': end_str, 'offset': offset},
                max_retries=3,
                sleep_time=40
            )

            if df is not None and not df.empty:
                print(f"成功获取 {len(df)} 条记录")
                page_data.append(df)
                offset += len(df)
                page += 1
                if len(df) < 5000:
                    print(f"  {start_str} ~ {end_str} 数据获取完成")
                    break
            else:
                print("失败或无数据")
                break

        if page_data:
            month_df = pd.concat(page_data, ignore_index=True)
            all_index.append(month_df)

            if len(all_index) >= 6:
                new_data = pd.concat(all_index, ignore_index=True)
                combined = pd.concat([existing_data, new_data], ignore_index=True)
                combined = combined.drop_duplicates(subset=['trade_date','ts_code'])
                combined.to_parquet(output_file, index=False)
                print(f"  已保存到 {output_file}")
                existing_data = combined
                all_index = []

            with open(progress_file, 'w') as f:
                f.write(end_str)
        else:
            print(f"  {start_str} ~ {end_str} 未获取到数据")

    # 保存剩余数据
    if all_index:
        new_data = pd.concat(all_index, ignore_index=True)
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        combined.to_parquet(output_file, index=False)
        print(f"\n已保存到 {output_file}")

    # 最终统计
    if output_file.exists():
        final_df = pd.read_parquet(output_file)
        print(f"\n总记录数: {len(final_df)}")
        if len(final_df) > 0:
            print(f"日期范围: {final_df['trade_date'].min()} ~ {final_df['trade_date'].max()}")
            print(f"概念数量: {final_df['ts_code'].nunique()}")


def find_start_date_from_2023(pro, start_year=2023) -> str | None:
    """从2023年起每年1月尝试，找到第一个有数据的交易日"""
    current_year = datetime.now().year
    for year in range(start_year, current_year + 1):
        print(f"\n检测 {year} 年1月是否有数据...")
        for day in range(1, 32):
            date_str = f"{year}01{day:02d}"
            try:
                df = pro.dc_index(trade_date=date_str)
                if df is not None and not df.empty:
                    print(f"  ✓ 找到起始数据日期: {date_str}")
                    return date_str
            except Exception as e:
                print(f"  {date_str} 请求失败: {e}")
            time.sleep(1)
        print(f"  {year} 年1月无数据，跳过")
    return None


def main():
    print("=" * 80)
    print("获取东方财富概念指数历史数据")
    print("=" * 80)

    # 设置Tushare token
    disable_proxy()
    token = get_tushare_token()
    pro = ts.pro_api(token)

    # 输出目录
    output_dir = Path(CONCEPTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取概念指数历史数据
    fetch_concept_index(pro, output_dir, start_date='20200101')

    print("\n" + "=" * 80)
    print("概念指数数据获取完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
