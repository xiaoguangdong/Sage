#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
获取10年期国债收益率（带断点续传和重试机制）

yc_cb接口单次最多返回2000条数据，需要多次调用
"""

import tushare as ts
import pandas as pd
import os
import time
from datetime import datetime, timedelta

from tushare_auth import get_tushare_token

def fetch_yield(token=None, curve_term=10, start_date='20200101', end_date='20251231'):
    """
    获取国债收益率（支持不同期限）

    yc_cb接口单次最多返回2000条数据，需要多次调用
    支持断点续传和重试机制

    Args:
        token: Tushare token
        curve_term: 期限（2=2年期，10=10年期）
        start_date: 开始日期
        end_date: 结束日期
    """
    pro = ts.pro_api(get_tushare_token(token))

    all_data = []
    offset = 0
    limit = 2000  # 单次最大2000条
    api_delay = 40  # 每次请求间隔40秒
    max_retries = 3  # 最大重试次数

    output_dir = 'data/tushare/macro'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'yield_{curve_term}y.parquet')

    # 断点续传：检查是否已有数据
    existing_dates = set()
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_parquet(filepath)
            if len(existing_df) > 0:
                existing_dates = set(existing_df['trade_date'].values)
                print(f"发现已有数据: {len(existing_df)}条，覆盖{len(existing_dates)}天")
                print(f"日期范围: {existing_df['trade_date'].min()} 至 {existing_df['trade_date'].max()}")
        except Exception as e:
            print(f"读取已有数据失败: {e}")

    print(f"开始获取{curve_term}年期国债收益率...")
    print(f"时间范围: {start_date} 至 {end_date}")
    print(f"每次请求间隔: {api_delay}秒，最大重试: {max_retries}次")

    batch_num = 0
    while True:
        print(f"  获取第 {offset + 1}-{offset + limit} 条（批次{batch_num + 1}）...", end=" ")

        retry_count = 0
        df = None
        
        while retry_count < max_retries:
            try:
                # 使用 curve_term 参数获取指定期限数据
                df = pro.yc_cb(curve_term=curve_term, limit=limit, offset=offset)
                print(f"OK ({len(df) if df is not None else 0}条)")
                break
            except Exception as e:
                retry_count += 1
                if "IP" in str(e) or "超限" in str(e):
                    print(f"IP限制触发（重试{retry_count}/{max_retries}）")
                    time.sleep(api_delay * retry_count)
                else:
                    print(f"错误: {e}")
                    break

        if df is None or len(df) == 0:
            if retry_count >= max_retries:
                print("  超过最大重试次数，停止获取")
            else:
                print("  没有更多数据，获取完成")
            break

        # 筛选指定期限
        df_y = df[df['curve_term'] == curve_term].copy()
        
        if len(df_y) > 0:
            # 过滤已存在的日期
            df_y = df_y[~df_y['trade_date'].isin(existing_dates)]
            
            if len(df_y) > 0:
                print(f"    新增{curve_term}年期: {len(df_y)}条")
                all_data.append(df_y)
                existing_dates.update(df_y['trade_date'].values)
            else:
                print(f"    都是重复数据")
        
        offset += limit
        batch_num += 1

        # 如果返回的数据少于limit，说明已经获取完
        if len(df) < limit:
            print("  已获取全部数据")
            break

        # 休息40秒
        print(f"    休息 {api_delay} 秒...")
        time.sleep(api_delay)
        
        # 每10批保存一次进度（断点续传）
        if batch_num % 10 == 0 and all_data:
            temp_df = pd.concat(all_data, ignore_index=True)
            temp_df = temp_df.sort_values('trade_date')
            temp_df.to_parquet(filepath, index=False)
            print(f"    已保存进度: {len(temp_df)}条")

    if all_data:
        # 合并所有数据
        result = pd.concat(all_data, ignore_index=True)

        # 筛选curve_type=0（注意curve_type可能是字符串）
        df_y = result[result['curve_type'].astype(str) == '0'][['trade_date', 'yield']].copy()
        df_y = df_y.sort_values('trade_date')

        # 筛选日期范围
        df_y = df_y[(df_y['trade_date'] >= start_date) & (df_y['trade_date'] <= end_date)]

        # 保存
        df_y.to_parquet(filepath, index=False)

        print(f"\n✓ {curve_term}年期国债收益率已保存: {filepath}")
        print(f"  总行数: {len(df_y)}")
        print(f"  日期范围: {df_y['trade_date'].min()} 至 {df_y['trade_date'].max()}")
        print(f"\n最新5条数据:")
        print(df_y.tail())

        return df_y
    else:
        print("未获取到新数据")
        return None


def fetch_yield_10y(token=None, start_date='20200101', end_date='20251231'):
    """获取10年期国债收益率"""
    return fetch_yield(token=token, curve_term=10, start_date=start_date, end_date=end_date)


def fetch_yield_2y(token=None, start_date='20200101', end_date='20251231'):
    """获取2年期国债收益率"""
    return fetch_yield(token=token, curve_term=2, start_date=start_date, end_date=end_date)


if __name__ == '__main__':
    # 获取10年期和2年期国债收益率
    print("=" * 60)
    fetch_yield_10y(start_date='20200101', end_date='20251231')
    print("\n" + "=" * 60)
    fetch_yield_2y(start_date='20200101', end_date='20251231')
