#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
整理北向资金行业数据（使用已有数据）
1. 使用已有的HS300成分股数据
2. 使用已有的个股北向资金持仓数据
3. 将个股数据映射到HS300指数成分股
4. 计算北向资金净流入
"""

import pandas as pd
from pathlib import Path

from scripts.data.macro.paths import CONSTITUENTS_DIR, NORTHBOUND_DIR

def load_data():
    """加载数据"""
    print("加载数据...")
    
    # 加载HS300成分股数据
    constituents_file = Path(CONSTITUENTS_DIR) / 'hs300_constituents_all.parquet'
    constituents_df = pd.read_parquet(constituents_file)
    print(f"  HS300成分股: {len(constituents_df)} 条记录")
    
    # 加载个股北向资金持仓数据
    holdings_file = Path(NORTHBOUND_DIR) / 'hk_hold.parquet'
    holdings_df = pd.read_parquet(holdings_file)
    print(f"  北向资金持仓: {len(holdings_df)} 条记录")
    
    return constituents_df, holdings_df


def merge_data(constituents_df, holdings_df):
    """合并数据"""
    print("\n合并数据...")
    
    # 合并持仓数据和成分股数据
    merged = holdings_df.merge(
        constituents_df[['con_code', 'index_code', 'weight']],
        left_on='ts_code',
        right_on='con_code',
        how='inner'
    )
    
    print(f"  匹配到 {len(merged)} 条记录")
    print(f"  覆盖股票数: {merged['ts_code'].nunique()}")
    
    return merged


def calculate_northbound_flow(merged_df):
    """计算北向资金净流入"""
    print("\n计算北向资金净流入...")
    
    if 'vol' not in merged_df.columns:
        print("  错误: 持仓数据中没有vol字段")
        return None
    
    # 按日期聚合计算净流入
    # 计算每日持仓变化量
    merged_df = merged_df.sort_values(['ts_code', 'trade_date'])
    merged_df['vol_change'] = merged_df.groupby('ts_code')['vol'].diff().fillna(0)
    
    # 按日期聚合
    daily_flow = merged_df.groupby('trade_date').agg({
        'vol_change': 'sum',
        'vol': 'sum',
        'ratio': 'mean'
    }).reset_index()
    
    daily_flow = daily_flow.rename(columns={
        'vol_change': 'net_flow',
        'vol': 'total_holdings',
        'ratio': 'avg_ratio'
    })
    
    print(f"  计算完成，共 {len(daily_flow)} 天的数据")
    
    return daily_flow


def calculate_by_stock(merged_df):
    """按股票统计北向资金"""
    print("\n按股票统计北向资金...")
    
    if 'vol' not in merged_df.columns:
        print("  错误: 持仓数据中没有vol字段")
        return None
    
    # 计算每日持仓变化量
    merged_df = merged_df.sort_values(['ts_code', 'trade_date'])
    merged_df['vol_change'] = merged_df.groupby('ts_code')['vol'].diff().fillna(0)
    
    # 按股票聚合
    stock_flow = merged_df.groupby('ts_code').agg({
        'vol_change': 'sum',
        'vol': 'last',
        'name': 'last',
        'weight': 'last'
    }).reset_index()
    
    stock_flow = stock_flow.rename(columns={
        'vol_change': 'net_flow',
        'vol': 'total_holdings'
    })
    
    stock_flow = stock_flow.sort_values('net_flow', ascending=False)
    
    print(f"  计算完成，共 {len(stock_flow)} 只股票")
    
    return stock_flow


def main():
    print("=" * 80)
    print("整理北向资金数据（使用已有数据）")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = Path(NORTHBOUND_DIR)
    
    # 1. 加载数据
    constituents_df, holdings_df = load_data()
    
    # 2. 合并数据
    merged_df = merge_data(constituents_df, holdings_df)
    
    # 3. 计算北向资金净流入
    daily_flow_df = calculate_northbound_flow(merged_df)
    if daily_flow_df is not None:
        daily_flow_file = output_dir / 'northbound_daily_flow_hs300.parquet'
        daily_flow_df.to_parquet(daily_flow_file, index=False)
        print(f"  已保存到 {daily_flow_file}")
    
    # 4. 按股票统计
    stock_flow_df = calculate_by_stock(merged_df)
    if stock_flow_df is not None:
        stock_flow_file = output_dir / 'northbound_stock_flow.parquet'
        stock_flow_df.to_parquet(stock_flow_file, index=False)
        print(f"  已保存到 {stock_flow_file}")
        
        # 显示统计信息
        print("\n" + "=" * 80)
        print("北向资金净流入TOP 10")
        print("=" * 80)
        print(stock_flow_df.head(10)[['ts_code', 'name', 'net_flow', 'total_holdings']].to_string(index=False))
        
        print("\n" + "=" * 80)
        print("北向资金净流出TOP 10")
        print("=" * 80)
        print(stock_flow_df.tail(10)[['ts_code', 'name', 'net_flow', 'total_holdings']].to_string(index=False))
    
    print("\n完成!")


if __name__ == '__main__':
    main()
