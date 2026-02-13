#!/usr/bin/env python3
# -u
# -*- coding: utf-8 -*-

"""
获取申万行业估值数据（PE、PB等）
从2020年开始每月获取一次，支持断点续传
"""

import pandas as pd
import tushare as ts
import time
import os
from datetime import datetime
from pathlib import Path

from tushare_auth import get_tushare_token

def get_sw_daily_with_retry(pro, start_date, end_date, offset=0, max_retries=3):
    """
    获取申万行业日线数据（所有行业），带重试机制

    Args:
        pro: Tushare pro对象
        start_date: 开始日期，格式YYYYMMDD
        end_date: 结束日期，格式YYYYMMDD
        offset: 分页偏移量
        max_retries: 最大重试次数

    Returns:
        DataFrame或None（失败时）
    """
    for attempt in range(max_retries):
        try:
            # 不传ts_code，获取所有行业数据，支持分页
            df = pro.sw_daily(start_date=start_date, end_date=end_date, offset=offset)
            time.sleep(45)  # 每次请求间隔45秒
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # 重试等待时间递增
                print(f"  请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"  请求失败，已达到最大重试次数: {e}")
                return None


def get_month_end_date(year, month):
    """获取某年某月的最后一天"""
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)

    last_day = next_month - pd.Timedelta(days=1)
    return last_day.strftime('%Y%m%d')


def get_months_range(start_year=2020, start_month=1, end_year=None, end_month=None):
    """生成日期月份范围"""
    if end_year is None:
        end_year = datetime.now().year
    if end_month is None:
        end_month = datetime.now().month

    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == start_year and month < start_month:
                continue
            if year == end_year and month > end_month:
                continue

            start_date = f"{year}{month:02d}01"
            end_date = get_month_end_date(year, month)
            months.append((start_date, end_date))

    return months


def load_progress(progress_file):
    """加载进度文件"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def save_progress(progress_file, completed_months):
    """保存进度文件"""
    with open(progress_file, 'w') as f:
        for month in sorted(completed_months):
            f.write(f"{month}\n")


def fetch_sw_valuation_data():
    """获取申万行业估值数据"""
    print("=" * 80)
    print("获取申万行业估值数据")
    print("=" * 80)

    pro = ts.pro_api(get_tushare_token())

    # 创建输出目录
    output_dir = Path('data/tushare/macro')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 进度文件
    progress_file = output_dir / 'sw_valuation_progress.txt'
    
    # 输出文件
    output_file = output_dir / 'sw_valuation.parquet'

    # 生成月份范围：2020年1月到现在
    months = get_months_range(start_year=2020, start_month=1)
    print(f"\n需要获取 {len(months)} 个月的数据")

    # 清空旧数据，重新获取
    print(f"\n清空旧数据，重新获取...")
    all_data = []
    # 删除进度文件
    if progress_file.exists():
        os.remove(progress_file)
    completed_months = set()
    print(f"  开始获取 {len(months)} 个月的数据")

    # 遍历每个月
    for i, (start_date, end_date) in enumerate(months):
        month_key = f"{start_date}_{end_date}"

        # 跳过已完成的月份
        if month_key in completed_months:
            print(f"\n[{i+1}/{len(months)}] {start_date} ~ {end_date} (已跳过)")
            continue

        print(f"\n[{i+1}/{len(months)}] 获取 {start_date} ~ {end_date} 的数据...")

        # 分页获取该月所有数据
        month_data = []
        offset = 0
        page = 1
        while True:
            print(f"  第{page}页获取 (offset={offset})...", end=' ')
            df = get_sw_daily_with_retry(pro, start_date, end_date, offset=offset, max_retries=3)
            
            if df is not None and not df.empty:
                print(f"成功获取 {len(df)} 条记录")
                month_data.append(df)
                offset += len(df)
                page += 1
                
                # 如果返回的记录数小于4000，说明没有更多数据了
                if len(df) < 4000:
                    print(f"  本月数据获取完成")
                    break
            else:
                print(f"失败")
                break

        # 保存当月数据
        if month_data:
            month_df = pd.concat(month_data, ignore_index=True)
            print(f"  本月共获取 {len(month_df)} 条记录")
            
            # 追加到总数据
            all_data.append(month_df)

            # 保存到文件
            if len(all_data) > 0:
                final_df = pd.concat(all_data, ignore_index=True)
                final_df.to_parquet(output_file, index=False)
                print(f"  已保存到 {output_file}")

            # 记录进度
            completed_months.add(month_key)
            save_progress(progress_file, completed_months)
        else:
            print(f"  本月未获取到数据")

    # 最终统计
    print("\n" + "=" * 80)
    print("数据获取完成")
    print("=" * 80)

    if os.path.exists(output_file):
        final_df = pd.read_parquet(output_file)
        print(f"\n总记录数: {len(final_df)}")
        print(f"数据范围: {final_df['trade_date'].min()} ~ {final_df['trade_date'].max()}")
        print(f"字段: {final_df.columns.tolist()}")
        print(f"\n数据预览:")
        print(final_df.head(10).to_string(index=False))

        # 按行业统计
        print(f"\n各行业数据量:")
        industry_stats = final_df.groupby('name').size().sort_values(ascending=False)
        for name, count in industry_stats.items():
            print(f"  {name}: {count} 条")


if __name__ == '__main__':
    fetch_sw_valuation_data()
