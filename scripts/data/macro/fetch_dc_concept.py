#!/usr/bin/env python3
# -u
# -*- coding: utf-8 -*-

"""
获取Tushare概念板块数据
使用dc_index、dc_member、dc_daily接口
"""

import pandas as pd
import tushare as ts
import time
from pathlib import Path

from tushare_auth import get_tushare_token
from scripts.data.macro.paths import CONCEPTS_DIR


def get_data_with_retry(pro, func_name, max_retries=3, **kwargs):
    """获取数据，带重试机制"""
    for attempt in range(max_retries):
        try:
            if func_name == 'dc_index':
                df = pro.dc_index(**kwargs)
            elif func_name == 'dc_member':
                df = pro.dc_member(**kwargs)
            elif func_name == 'dc_daily':
                df = pro.dc_daily(**kwargs)
            else:
                return None
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


def fetch_concept_data_new():
    """获取概念板块数据"""
    print("=" * 80)
    print("获取Tushare概念板块数据")
    print("=" * 80)

    # 设置Tushare token
    token = get_tushare_token()
    pro = ts.pro_api(token)

    # 创建输出目录
    output_dir = Path(CONCEPTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 获取概念指数数据
    print("\n[1/3] 获取概念指数数据...")
    df_index = get_data_with_retry(pro, 'dc_index')
    
    if df_index is not None and not df_index.empty:
        print(f"  成功获取 {len(df_index)} 条记录")
        
        # 保存
        index_file = output_dir / 'dc_index.parquet'
        df_index.to_parquet(index_file, index=False)
        print(f"  已保存到 {index_file}")
        
        # 统计
        print(f"  概念数量: {df_index['ts_code'].nunique()}")
        print(f"  日期范围: {df_index['trade_date'].min()} ~ {df_index['trade_date'].max()}")
    else:
        print("  获取失败")

    # 2. 获取概念成分股数据
    print("\n[2/3] 获取概念成分股数据...")
    df_member = get_data_with_retry(pro, 'dc_member')
    
    if df_member is not None and not df_member.empty:
        print(f"  成功获取 {len(df_member)} 条记录")
        
        # 保存
        member_file = output_dir / 'dc_member.parquet'
        df_member.to_parquet(member_file, index=False)
        print(f"  已保存到 {member_file}")
        
        # 统计
        print(f"  概念数量: {df_member['ts_code'].nunique()}")
        print(f"  成分股数量: {df_member['con_code'].nunique()}")
        print(f"  日期范围: {df_member['trade_date'].min()} ~ {df_member['trade_date'].max()}")
    else:
        print("  获取失败")

    # 3. 获取概念成分股日线数据（可选）
    print("\n[3/3] 获取概念成分股日线数据...")
    df_daily = get_data_with_retry(pro, 'dc_daily')
    
    if df_daily is not None and not df_daily.empty:
        print(f"  成功获取 {len(df_daily)} 条记录")
        
        # 保存
        daily_file = output_dir / 'dc_daily.parquet'
        df_daily.to_parquet(daily_file, index=False)
        print(f"  已保存到 {daily_file}")
        
        # 统计
        print(f"  概念数量: {df_daily['ts_code'].nunique()}")
        print(f"  日期范围: {df_daily['trade_date'].min()} ~ {df_daily['trade_date'].max()}")
    else:
        print("  获取失败（可能受IP限制）")

    print("\n" + "=" * 80)
    print("获取完成")
    print("=" * 80)


if __name__ == '__main__':
    fetch_concept_data_new()
