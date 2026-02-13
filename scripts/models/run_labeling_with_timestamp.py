#!/usr/bin/env python3
"""
运行市场状态打标 - 支持指定时间范围和时间戳

用法:
    python run_labeling_with_timestamp.py 2020-01-01 2026-02-09
    python run_labeling_with_timestamp.py 2018-01-01 2026-02-09
"""

import sys
import os
import shutil
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.models.label_hs300_daily_weekly import HS300Labeler
from scripts.data.tushare_suite import download_index_ohlc


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
    
    # 使用统一下载入口
    output_root = download_index_ohlc(
        start_date=start_date,
        end_date=end_date,
        output_dir=None,
        indices=[("000300.SH", "沪深300")],
        sleep_seconds=1,
    )

    src_file = output_root / "index_000300_SH_ohlc.parquet"
    if not src_file.exists():
        print("✗ 数据下载失败")
        return False

    legacy_dir = PROJECT_ROOT / "data" / "raw" / "tushare" / "index"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    output_file = legacy_dir / "index_000300_SH_ohlc.parquet"
    shutil.copy2(src_file, output_file)

    df = pd.read_parquet(output_file)
    print(f"✓ 成功获取 {len(df)} 条记录")
    if not df.empty and "date" in df.columns:
        print(f"  日期范围: {df['date'].min()} 至 {df['date'].max()}")
    print(f"✓ 数据已保存: {output_file}")
    return True


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
