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
import tushare as ts
import time
from pathlib import Path
from datetime import datetime


def get_sw_industry_constituents(pro):
    """获取申万行业成分股列表"""
    print("获取申万行业成分股列表...")
    
    # 获取所有申万一级行业
    industries = pro.index_classify(level='L1', src='SW2021')
    industries_l1 = industries[industries['index_code'].str.startswith('801')]
    
    print(f"  获取到 {len(industries_l1)} 个申万一级行业")
    
    constituents_list = []
    for i, row in industries_l1.iterrows():
        if (i + 1) % 5 == 0:
            print(f"  进度: {i + 1}/{len(industries_l1)}")
        
        try:
            df = pro.index_member(index_code=row['index_code'])
            if not df.empty:
                df['industry_code'] = row['industry_code']
                df['industry_name'] = row['industry_name']
                constituents_list.append(df)
            time.sleep(0.5)
        except Exception as e:
            print(f"  获取 {row['industry_name']} 成分股失败: {e}")
    
    if constituents_list:
        all_constituents = pd.concat(constituents_list, ignore_index=True)
        print(f"  共获取 {len(all_constituents)} 条成分股记录")
        return all_constituents
    else:
        return None


def get_stock_northbound_holding(pro, ts_codes, start_date, end_date):
    """获取个股北向资金持仓数据"""
    print(f"\n获取个股北向资金持仓 ({start_date} ~ {end_date})...")
    
    holdings_list = []
    total = len(ts_codes)
    
    for i, ts_code in enumerate(ts_codes):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i + 1}/{total}")
        
        try:
            df = pro.hk_hold(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if not df.empty:
                holdings_list.append(df)
            time.sleep(0.3)
        except Exception as e:
            if (i + 1) % 100 == 0:
                print(f"  获取 {ts_code} 失败: {e}")
    
    if holdings_list:
        all_holdings = pd.concat(holdings_list, ignore_index=True)
        print(f"  共获取 {len(all_holdings)} 条持仓记录")
        return all_holdings
    else:
        return None


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
    
    # 设置Tushare token
    token = '2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e'
    pro = ts.pro_api(token)
    
    # 创建输出目录
    output_dir = Path('data/tushare/northbound')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 获取申万行业成分股列表
    constituents_df = get_sw_industry_constituents(pro)
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
    unique_stocks = constituents_df['ts_code'].unique()
    print(f"\n需要获取 {len(unique_stocks)} 只股票的北向资金持仓数据")
    
    holdings_df = get_stock_northbound_holding(pro, unique_stocks, start_date, end_date)
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