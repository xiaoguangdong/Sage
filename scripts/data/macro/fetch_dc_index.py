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

from tushare_auth import get_tushare_token
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


def fetch_concept_index(pro, output_dir, start_date='20200101'):
    """获取概念指数历史数据"""
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
    end_dt = datetime.now() - timedelta(days=1)  # 昨天为止
    date_list = []
    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y%m%d')
        if date_str not in existing_dates:
            date_list.append(date_str)
        current_dt += timedelta(days=1)

    print(f"需要获取 {len(date_list)} 个交易日的概念指数数据")

    all_index = []
    total_dates = len(date_list)

    for i, trade_date in enumerate(date_list):
        print(f"\n[{i+1}/{total_dates}] 获取 {trade_date} 的概念指数数据...")

        offset = 0
        page = 1
        page_data = []

        while True:
            print(f"  第{page}页获取 (offset={offset})...", end=' ')
            df = get_with_retry(
                pro, 'dc_index',
                {'trade_date': trade_date, 'offset': offset},
                max_retries=3,
                sleep_time=40
            )

            if df is not None and not df.empty:
                print(f"成功获取 {len(df)} 条记录")
                page_data.append(df)
                offset += len(df)
                page += 1

                # dc_index单次最大5000条
                if len(df) < 5000:
                    print(f"  {trade_date} 数据获取完成")
                    break
            else:
                print(f"失败或无数据")
                break

        if page_data:
            date_df = pd.concat(page_data, ignore_index=True)
            all_index.append(date_df)
            existing_dates.add(trade_date)

            # 定期保存（每10天）
            if len(all_index) >= 10:
                new_data = pd.concat(all_index, ignore_index=True)
                combined = pd.concat([existing_data, new_data], ignore_index=True)
                combined.to_parquet(output_file, index=False)
                print(f"  已保存到 {output_file}")
                existing_data = combined
                all_index = []

            # 更新进度
            with open(progress_file, 'w') as f:
                f.write(trade_date)
        else:
            print(f"  {trade_date} 未获取到数据")

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


def main():
    print("=" * 80)
    print("获取东方财富概念指数历史数据")
    print("=" * 80)

    # 设置Tushare token
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
