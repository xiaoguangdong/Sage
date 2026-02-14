#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
整理北向资金行业数据
1. 获取申万行业成分股列表
2. 获取个股北向资金持仓数据
3. 将个股数据映射到行业
4. 计算行业级别的北向资金净流入
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.macro.paths import NORTHBOUND_DIR


def get_sw_industry_constituents():
    """读取申万行业成分股列表（来自统一下载器输出）"""
    print("读取申万行业成分股列表...")

    tushare_root = Path(NORTHBOUND_DIR).parent
    l1_path = tushare_root / "sw_industry" / "sw_industry_l1.parquet"
    member_path = tushare_root / "sw_industry" / "sw_index_member.parquet"

    if not l1_path.exists():
        print("  缺少 sw_industry_l1，请先运行 tushare_downloader.py --task sw_industry_classify")
        return None
    if not member_path.exists():
        print("  缺少 sw_index_member，请先运行 tushare_downloader.py --task sw_index_member")
        return None

    industries = pd.read_parquet(l1_path)
    members = pd.read_parquet(member_path)
    if industries.empty or members.empty:
        print("  行业分类或成分股数据为空")
        return None

    name_col = "industry_name" if "industry_name" in industries.columns else "index_name"
    industries = industries.rename(columns={name_col: "industry_name"})

    members = members.rename(columns={"con_code": "ts_code"})
    merged = members.merge(
        industries[["index_code", "industry_name", "industry_code"]] if "industry_code" in industries.columns else industries[["index_code", "industry_name"]],
        on="index_code",
        how="left",
    )
    print(f"  共读取 {len(merged)} 条成分股记录")
    return merged


def get_stock_northbound_holding(ts_codes, start_date, end_date):
    """读取个股北向资金持仓数据（来自统一下载器输出）"""
    print(f"\n读取个股北向资金持仓 ({start_date} ~ {end_date})...")
    hold_path = Path(NORTHBOUND_DIR) / "northbound_hold.parquet"
    if not hold_path.exists():
        print("  缺少 northbound_hold，请先运行 tushare_downloader.py --task northbound_hold")
        return None

    df = pd.read_parquet(hold_path)
    if df.empty:
        print("  northbound_hold 数据为空")
        return None

    df = df[df["ts_code"].isin(ts_codes)]
    df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
    print(f"  共读取 {len(df)} 条持仓记录")
    return df


def calculate_industry_northbound_flow(holdings_df, constituents_df):
    """计算行业级别的北向资金净流入"""
    print("\n计算行业北向资金净流入...")
    
    # 合并持仓数据和成分股数据
    merged = holdings_df.merge(
        constituents_df[['ts_code', 'industry_code', 'industry_name']],
        on='ts_code',
        how='left'
    )
    
    if 'vol' not in merged.columns:
        print("  错误: 持仓数据中没有vol字段")
        return None
    
    # 按行业和日期聚合
    industry_flow = merged.groupby(['industry_code', 'industry_name', 'trade_date']).agg({
        'vol': 'sum',
        'ratio': 'mean'
    }).reset_index()
    
    print(f"  计算完成，共 {len(industry_flow)} 条行业记录")
    return industry_flow


def main():
    print("=" * 80)
    print("整理北向资金行业数据")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = Path(NORTHBOUND_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 获取申万行业成分股列表
    constituents_df = get_sw_industry_constituents()
    if constituents_df is not None:
        constituents_file = output_dir / 'sw_constituents.parquet'
        constituents_df.to_parquet(constituents_file, index=False)
        print(f"  已保存到 {constituents_file}")
    else:
        print("  未能获取成分股数据，退出")
        return
    
    # 2. 获取个股北向资金持仓数据（最近3个月）
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now().replace(day=1) - pd.DateOffset(months=3)).strftime('%Y%m%d')
    
    # 获取所有成分股代码
    unique_stocks = constituents_df['ts_code'].dropna().unique()
    print(f"\n需要获取 {len(unique_stocks)} 只股票的北向资金持仓数据")
    
    holdings_df = get_stock_northbound_holding(unique_stocks, start_date, end_date)
    if holdings_df is not None:
        holdings_file = output_dir / 'stock_holdings.parquet'
        holdings_df.to_parquet(holdings_file, index=False)
        print(f"  已保存到 {holdings_file}")
    else:
        print("  未能获取持仓数据，退出")
        return
    
    # 3. 计算行业级别的北向资金净流入
    industry_flow_df = calculate_industry_northbound_flow(holdings_df, constituents_df)
    if industry_flow_df is not None:
        industry_flow_file = output_dir / 'industry_northbound_flow.parquet'
        industry_flow_df.to_parquet(industry_flow_file, index=False)
        print(f"  已保存到 {industry_flow_file}")
        
        # 显示统计信息
        print("\n" + "=" * 80)
        print("数据整理完成")
        print("=" * 80)
        print(f"\n行业北向资金流向统计:")
        print(industry_flow_df.groupby('industry_name').size().sort_values(ascending=False))
    
    print("\n完成!")


if __name__ == '__main__':
    main()
