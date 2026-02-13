#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
获取北向资金历史数据（带断点续传和重试机制）

moneyflow_hsgt: 每次最多300条，请求间隔30-45秒
hk_hold: 每次最多3800条，请求间隔30-45秒
hsgt_top10: 每次最多5000条，请求间隔30-45秒
"""

import tushare as ts
import pandas as pd
import os
import time
from datetime import datetime, timedelta

from tushare_auth import get_tushare_token


def fetch_northbound_flow(token=None, start_date='20200101', end_date='20251231'):
    """
    获取北向资金流向数据（moneyflow_hsgt）

    Args:
        token: Tushare token
        start_date: 开始日期
        end_date: 结束日期
    """
    pro = ts.pro_api(get_tushare_token(token))

    all_data = []
    offset = 0
    limit = 300  # 单次最大300条
    api_delay = 40  # 每次请求间隔40秒
    max_retries = 3  # 最大重试次数

    output_dir = 'data/tushare/northbound'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'daily_flow.parquet')

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

    print("开始获取北向资金流向数据...")
    print(f"时间范围: {start_date} 至 {end_date}")
    print(f"每次请求间隔: {api_delay}秒，最大重试: {max_retries}次")
    
    # 初始延迟，避免启动时立即触发IP限制
    print(f"等待 {api_delay} 秒后开始获取...")
    time.sleep(api_delay)

    batch_num = 0
    while True:
        print(f"  获取第 {offset + 1}-{offset + limit} 条（批次{batch_num + 1}）...", end=" ")

        retry_count = 0
        df = None
        
        while retry_count < max_retries:
            try:
                print(f"    发起请求...", end=" ", flush=True)
                df = pro.moneyflow_hsgt(start_date=start_date, end_date=end_date, limit=limit, offset=offset)
                print(f"收到响应", flush=True)
                print(f"OK ({len(df) if df is not None else 0}条)")
                break
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"错误: {e}（重试{retry_count}/{max_retries}）")
                    time.sleep(api_delay * retry_count)
                else:
                    print(f"错误: {e}，超过最大重试次数")
                    break

        if df is None or len(df) == 0:
            if retry_count >= max_retries:
                print("  超过最大重试次数，停止获取")
            else:
                print("  没有更多数据，获取完成")
            break

        # 过滤已存在的日期
        df_new = df[~df['trade_date'].isin(existing_dates)]
        
        if len(df_new) > 0:
            print(f"    新增: {len(df_new)}条")
            all_data.append(df_new)
            existing_dates.update(df_new['trade_date'].values)
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
        
        # 每5批保存一次进度
        if batch_num % 5 == 0 and all_data:
            temp_df = pd.concat(all_data, ignore_index=True)
            temp_df = temp_df.sort_values('trade_date')
            temp_df.to_parquet(filepath, index=False)
            print(f"    已保存进度: {len(temp_df)}条")

    if all_data:
        # 合并所有数据
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values('trade_date')
        
        # 保存
        result.to_parquet(filepath, index=False)

        print(f"\n✓ 北向资金流向数据已保存: {filepath}")
        print(f"  总行数: {len(result)}")
        print(f"  日期范围: {result['trade_date'].min()} 至 {result['trade_date'].max()}")
        print(f"\n最新5条数据:")
        print(result.tail())

        return result
    else:
        print("未获取到新数据")
        return None


def fetch_northbound_hold(token=None, start_date='20200101', end_date='20251231'):
    """
    获取北向资金持仓数据（hk_hold）

    Args:
        token: Tushare token
        start_date: 开始日期
        end_date: 结束日期
    """
    pro = ts.pro_api(get_tushare_token(token))

    all_data = []
    offset = 0
    limit = 3800  # 单次最大3800条
    api_delay = 40  # 每次请求间隔40秒
    max_retries = 3  # 最大重试次数

    output_dir = 'data/tushare/northbound'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'hk_hold.parquet')

    # 断点续传：检查是否已有数据
    existing_keys = set()
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_parquet(filepath)
            if len(existing_df) > 0:
                existing_keys = set(zip(existing_df['trade_date'].values, existing_df['ts_code'].values))
                print(f"发现已有数据: {len(existing_df)}条")
                print(f"日期范围: {existing_df['trade_date'].min()} 至 {existing_df['trade_date'].max()}")
        except Exception as e:
            print(f"读取已有数据失败: {e}")

    print("\n开始获取北向资金持仓数据（沪市SH + 深市SZ）...")
    print(f"时间范围: {start_date} 至 {end_date}")
    print(f"每次请求间隔: {api_delay}秒，最大重试: {max_retries}次")
    
    # 初始延迟
    print(f"等待 {api_delay} 秒后开始获取...")
    time.sleep(api_delay)

    # 分别获取沪市和深市数据
    exchanges = ['SH', 'SZ']
    
    for exchange in exchanges:
        print(f"\n=== 获取{exchange}交易所数据 ===")
        offset = 0
        batch_num = 0
        
        while True:
            print(f"  获取第 {offset + 1}-{offset + limit} 条（批次{batch_num + 1}）...", end=" ")

            retry_count = 0
            df = None
            
            while retry_count < max_retries:
                try:
                    df = pro.hk_hold(start_date=start_date, end_date=end_date, limit=limit, offset=offset, exchange=exchange)
                    print(f"OK ({len(df) if df is not None else 0}条)")
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"错误: {e}（重试{retry_count}/{max_retries}）")
                        time.sleep(api_delay * retry_count)
                    else:
                        print(f"错误: {e}，超过最大重试次数")
                        break

            if df is None or len(df) == 0:
                if retry_count >= max_retries:
                    print("  超过最大重试次数，停止获取")
                else:
                    print("  没有更多数据，获取完成")
                break

            # 过滤已存在的数据
            df_new = df[~df.apply(lambda x: (x['trade_date'], x['ts_code']) in existing_keys, axis=1)]
            
            if len(df_new) > 0:
                print(f"    新增: {len(df_new)}条")
                all_data.append(df_new)
                existing_keys.update(zip(df_new['trade_date'].values, df_new['ts_code'].values))
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
            
            # 每5批保存一次进度
            if batch_num % 5 == 0 and all_data:
                temp_df = pd.concat(all_data, ignore_index=True)
                temp_df = temp_df.sort_values(['trade_date', 'ts_code'])
                temp_df.to_parquet(filepath, index=False)
                print(f"    已保存进度: {len(temp_df)}条")

    if all_data:
        # 合并所有数据
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(['trade_date', 'ts_code'])
        
        # 保存
        result.to_parquet(filepath, index=False)

        print(f"\n✓ 北向资金持仓数据已保存: {filepath}")
        print(f"  总行数: {len(result)}")
        print(f"  日期范围: {result['trade_date'].min()} 至 {result['trade_date'].max()}")
        print(f"  股票数: {result['ts_code'].nunique()}")
        print(f"\n最新5条数据:")
        print(result.tail())

        return result
    else:
        print("未获取到新数据")
        return None


def fetch_northbound_top10(token=None, start_date='20200101', end_date='20251231'):
    """
    获取北向资金持仓TOP10数据（hsgt_top10）

    Args:
        token: Tushare token
        start_date: 开始日期
        end_date: 结束日期
    """
    pro = ts.pro_api(get_tushare_token(token))

    all_data = []
    offset = 0
    limit = 5000  # 单次最大5000条
    api_delay = 40  # 每次请求间隔40秒
    max_retries = 3  # 最大重试次数

    output_dir = 'data/tushare/northbound'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'hsgt_top10.parquet')

    # 断点续传：检查是否已有数据
    existing_keys = set()
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_parquet(filepath)
            if len(existing_df) > 0:
                existing_keys = set(zip(existing_df['trade_date'].values, existing_df['ts_code'].values))
                print(f"发现已有数据: {len(existing_df)}条")
                print(f"日期范围: {existing_df['trade_date'].min()} 至 {existing_df['trade_date'].max()}")
        except Exception as e:
            print(f"读取已有数据失败: {e}")

    print("\n开始获取北向资金持仓TOP10数据...")
    print(f"时间范围: {start_date} 至 {end_date}")
    print(f"每次请求间隔: {api_delay}秒，最大重试: {max_retries}次")
    
    # 初始延迟
    print(f"等待 {api_delay} 秒后开始获取...")
    time.sleep(api_delay)

    batch_num = 0
    while True:
        print(f"  获取第 {offset + 1}-{offset + limit} 条（批次{batch_num + 1}）...", end=" ")

        retry_count = 0
        df = None
        
        while retry_count < max_retries:
            try:
                df = pro.hsgt_top10(start_date=start_date, end_date=end_date, limit=limit, offset=offset)
                print(f"OK ({len(df) if df is not None else 0}条)")
                break
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"错误: {e}（重试{retry_count}/{max_retries}）")
                    time.sleep(api_delay * retry_count)
                else:
                    print(f"错误: {e}，超过最大重试次数")
                    break

        if df is None or len(df) == 0:
            if retry_count >= max_retries:
                print("  超过最大重试次数，停止获取")
            else:
                print("  没有更多数据，获取完成")
            break

        # 过滤已存在的数据
        df_new = df[~df.apply(lambda x: (x['trade_date'], x['ts_code']) in existing_keys, axis=1)]
        
        if len(df_new) > 0:
            print(f"    新增: {len(df_new)}条")
            all_data.append(df_new)
            existing_keys.update(zip(df_new['trade_date'].values, df_new['ts_code'].values))
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
        
        # 每5批保存一次进度
        if batch_num % 5 == 0 and all_data:
            temp_df = pd.concat(all_data, ignore_index=True)
            temp_df = temp_df.sort_values(['trade_date', 'ts_code'])
            temp_df.to_parquet(filepath, index=False)
            print(f"    已保存进度: {len(temp_df)}条")

    if all_data:
        # 合并所有数据
        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(['trade_date', 'ts_code'])
        
        # 保存
        result.to_parquet(filepath, index=False)

        print(f"\n✓ 北向资金持仓TOP10数据已保存: {filepath}")
        print(f"  总行数: {len(result)}")
        print(f"  日期范围: {result['trade_date'].min()} 至 {result['trade_date'].max()}")
        print(f"  股票数: {result['ts_code'].nunique()}")
        print(f"\n最新5条数据:")
        print(result.tail())

        return result
    else:
        print("未获取到新数据")
        return None


if __name__ == '__main__':
    import sys
    
    # 检查命令行参数，决定获取哪个数据
    if len(sys.argv) > 1:
        data_type = sys.argv[1]
        if data_type == 'flow':
            fetch_northbound_flow(start_date='20200101', end_date='20251231')
        elif data_type == 'hold':
            fetch_northbound_hold(start_date='20200101', end_date='20251231')
        elif data_type == 'top10':
            fetch_northbound_top10(start_date='20200101', end_date='20251231')
        else:
            print(f"未知数据类型: {data_type}")
            print("用法: python3 fetch_northbound_history.py [flow|hold|top10]")
    else:
        # 默认只获取资金流向数据
        print("用法: python3 fetch_northbound_history.py [flow|hold|top10]")
        print("示例:")
        print("  python3 fetch_northbound_history.py flow   # 获取北向资金流向")
        print("  python3 fetch_northbound_history.py hold   # 获取北向资金持仓")
        print("  python3 fetch_northbound_history.py top10  # 获取北向资金持仓TOP10")
