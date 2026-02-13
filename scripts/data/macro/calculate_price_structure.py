#!/usr/bin/env python3
# -u
# -*- coding: utf-8 -*-

"""
计算个股价格结构指标
- 相对强度
- 动量（累计涨幅）
- 位置（当前价格在历史区间中的位置）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from scripts.data.macro.paths import DAILY_DIR, PRICE_STRUCTURE_DIR, TUSHARE_ROOT

def calculate_price_structure():
    """计算个股价格结构指标"""
    print("=" * 80)
    print("计算个股价格结构指标")
    print("=" * 80)

    # 加载个股日度数据
    print("\n加载个股日度数据...")
    daily_dir = Path(DAILY_DIR)
    daily_files = sorted(daily_dir.glob('daily_*.parquet'))
    
    all_daily = []
    for f in daily_files:
        df = pd.read_parquet(f)
        all_daily.append(df)
    
    daily_df = pd.concat(all_daily, ignore_index=True)
    print(f"  加载 {len(daily_df)} 条记录")
    print(f"  日期范围: {daily_df['trade_date'].min()} ~ {daily_df['trade_date'].max()}")

    # 加载指数数据（沪深300）
    print("\n加载指数数据...")
    index_df = pd.read_parquet(str(TUSHARE_ROOT / 'index' / 'index_000300_SH_ohlc.parquet'))
    index_df = index_df.rename(columns={'date': 'trade_date', 'pct_change': 'pct_chg'})
    print(f"  加载 {len(index_df)} 条记录")
    print(f"  日期范围: {index_df['trade_date'].min()} ~ {index_df['trade_date'].max()}")

    # 输出文件
    output_dir = Path(PRICE_STRUCTURE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'price_structure.parquet'

    # 按股票分组计算指标
    print("\n开始计算价格结构指标...")
    
    results = []
    stock_count = 0
    total_stocks = daily_df['ts_code'].nunique()
    
    for ts_code, stock_data in daily_df.groupby('ts_code'):
        stock_count += 1
        if stock_count % 100 == 0:
            print(f"  处理进度: {stock_count}/{total_stocks} ({stock_count/total_stocks*100:.1f}%)")
        
        # 按日期排序
        stock_data = stock_data.sort_values('trade_date').copy()
        
        # 计算动量（20日累计涨幅）
        stock_data['momentum_20d'] = stock_data['pct_chg'].rolling(20).apply(lambda x: (1 + x/100).prod() - 1)
        stock_data['momentum_60d'] = stock_data['pct_chg'].rolling(60).apply(lambda x: (1 + x/100).prod() - 1)
        
        # 计算位置（当前价格在60日区间中的位置）
        stock_data['position_60d'] = stock_data['close'].rolling(60).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
        
        # 计算相对强度（与沪深300对比）
        stock_data = stock_data.merge(
            index_df[['trade_date', 'pct_chg']],
            on='trade_date',
            how='left',
            suffixes=('', '_index')
        )
        
        # 相对强度 = 个股涨幅 - 指数涨幅
        stock_data['relative_strength_20d'] = stock_data['momentum_20d'] - stock_data['pct_chg_index'].rolling(20).apply(
            lambda x: (1 + x/100).prod() - 1
        )
        
        # 添加股票代码
        stock_data['ts_code'] = ts_code
        
        results.append(stock_data)
    
    # 合并所有结果
    print("\n合并结果...")
    final_df = pd.concat(results, ignore_index=True)
    
    # 保存
    final_df = final_df[['ts_code', 'trade_date', 'momentum_20d', 'momentum_60d', 'position_60d', 'relative_strength_20d']]
    final_df.to_parquet(output_file, index=False)
    print(f"  已保存到 {output_file}")
    
    # 统计
    print("\n" + "=" * 80)
    print("计算完成")
    print("=" * 80)
    print(f"\n总记录数: {len(final_df)}")
    print(f"股票数量: {final_df['ts_code'].nunique()}")
    print(f"日期范围: {final_df['trade_date'].min()} ~ {final_df['trade_date'].max()}")
    
    # 数据质量检查
    print(f"\n数据质量检查:")
    for col in ['momentum_20d', 'momentum_60d', 'position_60d', 'relative_strength_20d']:
        non_null = final_df[col].notna().sum()
        print(f"  {col}: {non_null}/{len(final_df)} ({non_null/len(final_df)*100:.1f}%)")
    
    print(f"\n数据预览:")
    print(final_df.head(10).to_string(index=False))


if __name__ == '__main__':
    calculate_price_structure()
