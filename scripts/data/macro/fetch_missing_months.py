#!/usr/bin/env python3
# -u
# -*- coding: utf-8 -*-

"""
补充遗漏的申万行业估值数据
"""

import pandas as pd
import tushare as ts
import time
from pathlib import Path

from tushare_auth import get_tushare_token


def get_sw_daily_with_retry(pro, start_date, end_date, offset=0, max_retries=3):
    """获取申万行业日线数据，带重试机制"""
    for attempt in range(max_retries):
        try:
            df = pro.sw_daily(start_date=start_date, end_date=end_date, offset=offset)
            time.sleep(45)
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


def fetch_missing_months():
    """补充遗漏的月份"""
    print("=" * 80)
    print("补充遗漏的申万行业估值数据")
    print("=" * 80)

    # 设置Tushare token
    token = get_tushare_token()
    pro = ts.pro_api(token)

    # 遗漏的月份
    missing_months = [
        ('20200401', '20200430'),
        ('20201101', '20201130'),
        ('20221101', '20221130')
    ]

    print(f"\n需要补充 {len(missing_months)} 个月的数据:")
    for start, end in missing_months:
        print(f"  {start} ~ {end}")

    # 输出文件
    output_dir = Path('data/tushare/macro')
    output_file = output_dir / 'sw_valuation.parquet'

    # 加载已有数据
    print(f"\n加载已有数据...")
    existing_data = pd.read_parquet(output_file)
    print(f"  已有 {len(existing_data)} 条记录")

    # 遍历遗漏的月份
    for i, (start_date, end_date) in enumerate(missing_months):
        print(f"\n[{i+1}/{len(missing_months)}] 获取 {start_date} ~ {end_date} 的数据...")

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

            # 合并到已有数据
            combined_data = pd.concat([existing_data, month_df], ignore_index=True)
            combined_data.to_parquet(output_file, index=False)
            print(f"  已保存到 {output_file}")

            # 更新已有数据
            existing_data = combined_data
        else:
            print(f"  本月未获取到数据")

    # 最终统计
    print("\n" + "=" * 80)
    print("数据补充完成")
    print("=" * 80)

    final_df = pd.read_parquet(output_file)
    print(f"\n总记录数: {len(final_df)}")
    print(f"数据范围: {final_df['trade_date'].min()} ~ {final_df['trade_date'].max()}")

    # 按月统计
    final_df['month'] = final_df['trade_date'].apply(lambda x: x[:6])
    month_counts = final_df['month'].value_counts().sort_index()
    print(f"\n按月统计: {len(month_counts)} 个月")


if __name__ == '__main__':
    fetch_missing_months()
