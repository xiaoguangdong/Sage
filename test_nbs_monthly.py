#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用NBS固定资产投资数据进行月度测试

按月份分组输出机会行业
"""

import pandas as pd
import json
import yaml
from datetime import datetime
import os


def load_fai_sw_data():
    """加载对齐后的申万行业FAI数据"""
    fai_file = 'data/processed/fai_sw_industry.parquet'
    if not os.path.exists(fai_file):
        print(f"文件不存在: {fai_file}")
        print("请先运行 align_fai_to_sw.py 生成数据")
        return None
    
    df = pd.read_parquet(fai_file)
    print(f"加载FAI数据: {len(df)}条记录")
    print(f"申万行业: {df['sw_industry'].nunique()}个")
    print(f"时间范围: {df['date'].min()} ~ {df['date'].max()}")
    
    return df


def load_macro_data():
    """加载宏观数据"""
    macro_data = []
    
    # 加载CPI
    cpi_file = 'data/tushare/macro/tushare_cpi.parquet'
    if os.path.exists(cpi_file):
        cpi = pd.read_parquet(cpi_file)
        cpi['date'] = pd.to_datetime(cpi['month'].astype(str), format='%Y%m')
        cpi = cpi.rename(columns={'nt_yoy': 'cpi_yoy'})
        macro_data.append(cpi[['date', 'cpi_yoy']])
    
    # 加载PPI
    ppi_file = 'data/tushare/macro/tushare_ppi.parquet'
    if os.path.exists(ppi_file):
        ppi = pd.read_parquet(ppi_file)
        ppi['date'] = pd.to_datetime(ppi['month'].astype(str), format='%Y%m')
        ppi = ppi.rename(columns={'ppi_yoy': 'ppi_yoy'})
        macro_data.append(ppi[['date', 'ppi_yoy']])
    
    # 加载PMI
    pmi_file = 'data/tushare/macro/tushare_pmi.parquet'
    if os.path.exists(pmi_file):
        pmi = pd.read_parquet(pmi_file)
        pmi['date'] = pd.to_datetime(pmi['MONTH'].astype(str), format='%Y%m')
        pmi = pmi.rename(columns={'PMI010000': 'pmi'})
        macro_data.append(pmi[['date', 'pmi']])
    
    # 加载收益率
    yield_file = 'data/tushare/macro/yield_10y.parquet'
    if os.path.exists(yield_file):
        yield_10y = pd.read_parquet(yield_file)
        yield_10y['date'] = pd.to_datetime(yield_10y['trade_date'])
        yield_10y = yield_10y.rename(columns={'yield': 'yield_10y'})
        macro_data.append(yield_10y[['date', 'yield_10y']])
    
    if not macro_data:
        return None
    
    # 合并宏观数据
    macro = macro_data[0]
    for df in macro_data[1:]:
        macro = macro.merge(df, on='date', how='outer')
    
    # 按月聚合
    macro['year_month'] = macro['date'].dt.to_period('M')
    macro_monthly = macro.groupby('year_month').agg({
        'cpi_yoy': 'last',
        'ppi_yoy': 'last',
        'pmi': 'last',
        'yield_10y': 'last'
    }).reset_index()
    macro_monthly['date'] = macro_monthly['year_month'].dt.to_timestamp()
    
    print(f"\n加载宏观数据: {len(macro_monthly)}个月")
    print(f"时间范围: {macro_monthly['date'].min()} ~ {macro_monthly['date'].max()}")
    
    return macro_monthly


def analyze_monthly(fai_df, macro_df):
    """
    按月分析机会行业
    
    Args:
        fai_df: 申万行业FAI数据
        macro_df: 宏观数据
    """
    print("\n" + "=" * 80)
    print("按月分析机会行业")
    print("=" * 80)
    
    # 合并数据
    fai_df['year_month'] = fai_df['date'].dt.to_period('M')
    fai_monthly = fai_df.groupby(['year_month', 'sw_industry']).agg({
        'fai_yoy': 'last',
        'fai_mom': 'last'
    }).reset_index()
    fai_monthly['date'] = fai_monthly['year_month'].dt.to_timestamp()
    
    merged = fai_monthly.merge(macro_df, on='date', how='left')
    
    # 筛选2024年9月到2025年12月的数据
    start_date = pd.to_datetime('2024-09-01')
    end_date = pd.to_datetime('2025-12-31')
    merged = merged[(merged['date'] >= start_date) & (merged['date'] <= end_date)]
    
    print(f"\n分析时间段: {start_date} ~ {end_date}")
    print(f"总月份数: {merged['date'].nunique()}个月")
    
    # 按月分析
    for month in sorted(merged['date'].unique()):
        month_data = merged[merged['date'] == month]
        
        print(f"\n{'=' * 80}")
        print(f"月份: {month.strftime('%Y年%m月')}")
        print(f"{'=' * 80}")
        
        # 宏观环境
        macro_row = month_data.iloc[0]
        print(f"\n宏观环境:")
        print(f"  CPI同比: {macro_row['cpi_yoy']:.2f}%")
        print(f"  PPI同比: {macro_row['ppi_yoy']:.2f}%")
        print(f"  PMI: {macro_row['pmi']:.2f}")
        print(f"  10年期国债收益率: {macro_row['yield_10y']:.2f}%")
        
        # 系统风险判断
        systemic_risk = False
        if macro_row['pmi'] < 48.5:
            systemic_risk = True
            print(f"\n⚠️  系统风险: PMI低于阈值(48.5)")
        
        if systemic_risk:
            print("\n系统风险期间，建议降低仓位")
            continue
        
        # 行业分析
        print(f"\n行业投资增速分析:")
        
        # 投资扩张行业（FAI同比增长>0）
        expansion = month_data[month_data['fai_yoy'] > 0].sort_values('fai_yoy', ascending=False)
        
        if len(expansion) > 0:
            print(f"\n  🚀 投资扩张行业 ({len(expansion)}个):")
            for i, row in expansion.head(10).iterrows():
                momentum = "↑" if row['fai_mom'] > 0 else "↓"
                print(f"    {i+1}. {row['sw_industry']:12s} 同比+{row['fai_yoy']:6.2f}% 环比{momentum}{abs(row['fai_mom']):5.2f}%")
        else:
            print("\n  无投资扩张行业")
        
        # 投资萎缩行业（FAI同比增长<0）
        contraction = month_data[month_data['fai_yoy'] < 0].sort_values('fai_yoy')
        
        if len(contraction) > 0:
            print(f"\n  📉 投资萎缩行业 ({len(contraction)}个):")
            for i, row in contraction.head(5).iterrows():
                print(f"    {i+1}. {row['sw_industry']:12s} 同比{row['fai_yoy']:6.2f}%")
        
        # 机会行业推荐（基于综合评分）
        print(f"\n  💰 机会行业推荐:")
        
        opportunity_industries = []
        
        for idx, row in month_data.iterrows():
            # 综合评分 = FAI同比增速权重 + 宏观环境因子
            score = 0
            
            # FAI增速（权重60%）
            if row['fai_yoy'] > 0:
                score += min(row['fai_yoy'] * 0.6, 60)
            
            # 环比动量（权重20%）
            if row['fai_mom'] > 0:
                score += min(row['fai_mom'] * 0.2, 20)
            
            # 宏观环境因子（权重20%）
            if macro_row['pmi'] > 50:
                score += 10
            if macro_row['ppi_yoy'] > 0:
                score += 10
            
            opportunity_industries.append({
                'industry': row['sw_industry'],
                'fai_yoy': row['fai_yoy'],
                'fai_mom': row['fai_mom'],
                'score': score
            })
        
        # 按评分排序
        opportunity_industries.sort(key=lambda x: x['score'], reverse=True)
        
        # 输出TOP 10
        top_opportunities = [x for x in opportunity_industries if x['score'] > 0]
        
        if len(top_opportunities) > 0:
            for i, item in enumerate(top_opportunities[:10], 1):
                print(f"    {i}. {item['industry']:12s} 评分:{item['score']:5.1f}分  FAI同比:+{item['fai_yoy']:5.2f}%")
        else:
            print("    暂无机会行业")
    
    # 统计各行业出现频率
    print(f"\n{'=' * 80}")
    print(f"机会行业出现频率统计")
    print(f"{'=' * 80}")
    
    industry_frequency = {}
    for month in sorted(merged['date'].unique()):
        month_data = merged[merged['date'] == month]
        
        # 获取TOP 5机会行业
        month_opportunities = []
        for idx, row in month_data.iterrows():
            score = 0
            if row['fai_yoy'] > 0:
                score += min(row['fai_yoy'] * 0.6, 60)
            if row['fai_mom'] > 0:
                score += min(row['fai_mom'] * 0.2, 20)
            if row['pmi'] > 50:
                score += 10
            if row['ppi_yoy'] > 0:
                score += 10
            
            month_opportunities.append({
                'industry': row['sw_industry'],
                'score': score
            })
        
        month_opportunities.sort(key=lambda x: x['score'], reverse=True)
        top5 = [x['industry'] for x in month_opportunities[:5] if x['score'] > 0]
        
        for industry in top5:
            if industry not in industry_frequency:
                industry_frequency[industry] = 0
            industry_frequency[industry] += 1
    
    # 按频率排序
    sorted_frequency = sorted(industry_frequency.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTOP 15高频机会行业:")
    for i, (industry, count) in enumerate(sorted_frequency[:15], 1):
        percentage = count / merged['date'].nunique() * 100
        print(f"  {i:2d}. {industry:12s} 出现{count:2d}次 ({percentage:5.1f}%)")


def main():
    """主函数"""
    print("=" * 80)
    print("NBS固定资产投资数据月度分析")
    print("=" * 80)
    
    # 加载数据
    fai_df = load_fai_sw_data()
    if fai_df is None:
        return
    
    macro_df = load_macro_data()
    if macro_df is None:
        print("无法加载宏观数据")
        return
    
    # 分析
    analyze_monthly(fai_df, macro_df)
    
    print("\n" + "=" * 80)
    print("分析完成")
    print("=" * 80)


if __name__ == '__main__':
    main()