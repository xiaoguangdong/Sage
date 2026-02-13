#!/usr/bin/env python3
"""
运行市场状态打标 - 支持指定时间范围和时间戳

用法:
    python run_labeling_with_timestamp.py 2020-01-01 2026-02-09
    python run_labeling_with_timestamp.py 2018-01-01 2026-02-09
"""

import sys
import os
import tushare as ts
import pandas as pd
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.label_hs300_daily_weekly import HS300Labeler


def download_index_data(start_date, end_date):
    """
    下载指定时间段的指数数据
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    """
    print("=" * 80)
    print(f"下载指数数据: {start_date} 至 {end_date}")
    print("=" * 80)
    
    # 初始化Tushare
    ts_pro = ts.pro_api('2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e')
    
    # 下载沪深300数据
    df = ts_pro.index_daily(
        ts_code='000300.SH',
        start_date=start_date.replace('-', ''),
        end_date=end_date.replace('-', '')
    )
    
    if df is not None and not df.empty:
        print(f"✓ 成功获取 {len(df)} 条记录")
        print(f"  日期范围: {df['trade_date'].min()} 至 {df['trade_date'].max()}")
        
        # 重命名列
        df = df.rename(columns={
            'trade_date': 'date',
            'ts_code': 'code',
            'pct_chg': 'pct_change'
        })
        
        # 保存数据
        output_dir = "data/tushare/index"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "index_000300_SH_ohlc.parquet")
        df.to_parquet(output_file)
        
        print(f"✓ 数据已保存: {output_file}")
        return True
    else:
        print("✗ 数据下载失败")
        return False


def run_labeling(start_date, end_date):
    """
    运行打标流程（日线和周线）
    
    Args:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    """
    print("\n" + "="*80)
    print(f"开始打标流程: {start_date} 至 {end_date}")
    print("="*80)
    
    # 1. 下载数据
    success = download_index_data(start_date, end_date)
    if not success:
        print("数据下载失败，退出")
        return
    
    # 2. 运行日线打标
    print("\n" + "="*80)
    print("测试日线模式")
    print("="*80)
    labeler_daily = HS300Labeler(timeframe='daily')
    labeler_daily.run()
    
    # 3. 运行周线打标
    print("\n" + "="*80)
    print("测试周线模式")
    print("="*80)
    labeler_weekly = HS300Labeler(timeframe='weekly')
    labeler_weekly.run()
    
    print("\n" + "="*80)
    print("✓ 完成！")
    print("="*80)


def main():
    """主函数"""
    # 从命令行参数获取日期
    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    else:
        print("使用默认日期范围: 2020-01-01 至 2026-02-09")
        start_date = "2020-01-01"
        end_date = "2026-02-09"
    
    # 运行打标
    run_labeling(start_date, end_date)


if __name__ == "__main__":
    main()