#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AI相关行业投资增速分析

分析数据中心、芯片产能、服务器等AI相关行业的投资情况
"""

import pandas as pd
import json
from datetime import datetime
import os


def parse_nbs_json(json_file: str, target_codes: list) -> pd.DataFrame:
    """解析NBS JSON数据，提取目标代码的数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'returndata' not in data:
        return pd.DataFrame()
    
    returndata = data['returndata']
    
    # 获取维度节点
    zb_nodes = []
    sj_nodes = []
    
    for node in returndata.get('wdnodes', []):
        if node['wdcode'] == 'zb':
            zb_nodes = node['nodes']
        elif node['wdcode'] == 'sj':
            sj_nodes = node['nodes']
    
    # 创建映射
    zb_code_to_name = {node['code']: node['name'] for node in zb_nodes}
    sj_code_to_name = {node['code']: node['name'] for node in sj_nodes}
    
    # 解析数据节点
    records = []
    for datanode in returndata.get('datanodes', []):
        if not datanode['data']['hasdata']:
            continue
        
        wds = datanode['wds']
        zb_code = None
        sj_code = None
        
        for wd in wds:
            if wd['wdcode'] == 'zb':
                zb_code = wd['valuecode']
            elif wd['wdcode'] == 'sj':
                sj_code = wd['valuecode']
        
        # 只提取目标代码
        if zb_code and sj_code and zb_code in target_codes:
            value = datanode['data']['data']
            records.append({
                'fai_code': zb_code,
                'fai_name': zb_code_to_name.get(zb_code, ''),
                'time_code': sj_code,
                'time_name': sj_code_to_name.get(sj_code, ''),
                'fai_yoy': value
            })
    
    df = pd.DataFrame(records)
    
    # 转换时间
    if len(df) > 0:
        df['date'] = pd.to_datetime(df['time_code'].astype(str), format='%Y%m')
    
    return df


def analyze_ai_investment():
    """分析AI相关投资"""
    print("=" * 80)
    print("AI相关行业投资增速分析")
    print("=" * 80)
    
    # AI相关的固定资产投资领域代码
    ai_fai_codes = [
        'A040314',  # 计算机、通信和其他电子设备制造业
        'A04031M',   # 信息传输、软件和信息技术服务业
        'A040310',   # 专用设备制造业（包含服务器等设备）
        'A04030Z',   # 通用设备制造业
        'A040313',   # 电气机械和器材制造业
    ]
    
    # 加载FAI数据
    fai_file = 'data/tushare/macro/A0403_固定资产投资.json'
    if not os.path.exists(fai_file):
        print(f"文件不存在: {fai_file}")
        return
    
    fai_df = parse_nbs_json(fai_file, ai_fai_codes)
    
    if len(fai_df) == 0:
        print("未找到AI相关投资数据")
        return
    
    print(f"\n找到AI相关投资数据: {len(fai_df)}条记录")
    print(f"涉及领域: {fai_df['fai_code'].nunique()}个")
    print(f"时间范围: {fai_df['date'].min()} ~ {fai_df['date'].max()}")
    
    # 筛选2024年9月到2025年12月的数据
    fai_df = fai_df[(fai_df['date'] >= '2024-09-01') & (fai_df['date'] <= '2025-12-31')]
    
    print(f"\n分析时间段: 2024-09 ~ 2025-12")
    print(f"数据点: {len(fai_df)}个")
    
    # 按领域分析
    print("\n" + "=" * 80)
    print("各AI相关领域投资增速分析")
    print("=" * 80)
    
    for fai_code in fai_df['fai_code'].unique():
        code_data = fai_df[fai_df['fai_code'] == fai_code].sort_values('date')
        
        if len(code_data) < 2:
            continue
        
        name = code_data['fai_name'].iloc[0]
        
        print(f"\n【{name}】({fai_code})")
        print(f"  数据点: {len(code_data)}个")
        
        # 最新数据
        latest = code_data.iloc[-1]
        print(f"  最新({latest['date'].strftime('%Y-%m')}): 同比{latest['fai_yoy']:+.2f}%")
        
        # 趋势分析
        if len(code_data) >= 3:
            recent_3 = code_data.tail(3)
            avg_3m = recent_3['fai_yoy'].mean()
            print(f"  近3月平均: {avg_3m:+.2f}%")
            
            # 检查趋势
            if avg_3m > 10:
                print(f"  状态: 🚀 高景气（投资活跃）")
            elif avg_3m > 0:
                print(f"  状态: ✅ 温和增长")
            elif avg_3m > -5:
                print(f"  状态: ⚠️  增速放缓")
            else:
                print(f"  状态: 📉 投资萎缩")
            
            # 检查是否连续扩张
            if (recent_3['fai_yoy'] > 0).all():
                print(f"  信号: 连续3个月投资扩张！")
            
            # 环比动量
            if len(recent_3) >= 2:
                mom_change = recent_3['fai_yoy'].iloc[-1] - recent_3['fai_yoy'].iloc[-2]
                if mom_change > 0:
                    print(f"  动量: ↑ 加速中（环比+{mom_change:.2f}%）")
                else:
                    print(f"  动量: ↓ 减速中（环比{mom_change:.2f}%）")
        
        # 历史对比
        if len(code_data) >= 6:
            first = code_data.iloc[0]
            last = code_data.iloc[-1]
            change = last['fai_yoy'] - first['fai_yoy']
            print(f"  期间变化: {first['date'].strftime('%Y-%m')}({first['fai_yoy']:+.2f}%) → {last['date'].strftime('%Y-%m')}({last['fai_yoy']:+.2f}%), 变化{change:+.2f}%")
    
    # 综合分析
    print("\n" + "=" * 80)
    print("AI行业综合分析")
    print("=" * 80)
    
    # 按月汇总
    monthly_summary = fai_df.groupby('date').agg({
        'fai_yoy': 'mean',
        'fai_code': 'count'
    }).reset_index()
    monthly_summary.columns = ['date', 'avg_fai_yoy', 'industry_count']
    
    print("\nAI行业整体投资增速:")
    for _, row in monthly_summary.sort_values('date').iterrows():
        print(f"  {row['date'].strftime('%Y-%m')}: 平均{row['avg_fai_yoy']:+.2f}%, 涉及{row['industry_count']}个领域")
    
    # 最新综合评估
    latest_summary = monthly_summary.iloc[-1]
    print(f"\n最新评估（{latest_summary['date'].strftime('%Y-%m')}）:")
    print(f"  AI行业平均投资增速: {latest_summary['avg_fai_yoy']:+.2f}%")
    
    if latest_summary['avg_fai_yoy'] > 15:
        print(f"  结论: 🔥 AI行业投资非常活跃，景气度高")
    elif latest_summary['avg_fai_yoy'] > 5:
        print(f"  结论: ✅ AI行业投资稳步增长，景气度良好")
    elif latest_summary['avg_fai_yoy'] > 0:
        print(f"  结论: ⚠️  AI行业投资温和增长，景气度一般")
    else:
        print(f"  结论: 📉 AI行业投资萎缩，景气度低")
    
    # 与宏观经济对比
    print("\n" + "=" * 80)
    print("与宏观经济对比")
    print("=" * 80)
    
    # 加载宏观数据
    pmi_file = 'data/tushare/macro/tushare_pmi.parquet'
    if os.path.exists(pmi_file):
        pmi = pd.read_parquet(pmi_file)
        pmi['date'] = pd.to_datetime(pmi['MONTH'].astype(str), format='%Y%m')
        
        latest_pmi = pmi[pmi['date'] == latest_summary['date']]
        if len(latest_pmi) > 0:
            pmi_value = latest_pmi['PMI010000'].iloc[0]
            print(f"\n  宏观PMI: {pmi_value:.2f}")
            print(f"  AI投资增速: {latest_summary['avg_fai_yoy']:+.2f}%")
            
            if latest_summary['avg_fai_yoy'] > 0 and pmi_value < 50:
                print(f"  结论: 🌟 AI行业逆势增长，结构性机会突出！")
            elif latest_summary['avg_fai_yoy'] > 0 and pmi_value > 50:
                print(f"  结论: 📈 AI行业与宏观经济共振向上")
            else:
                print(f"  结论: 📊 AI行业跟随宏观经济趋势")
    
    # 投资建议
    print("\n" + "=" * 80)
    print("投资建议")
    print("=" * 80)
    
    # 找出表现最好的AI领域
    latest_data = fai_df[fai_df['date'] == fai_df['date'].max()]
    top_ai = latest_data.sort_values('fai_yoy', ascending=False).head(3)
    
    print(f"\n当前最值得关注的AI领域:")
    for i, row in top_ai.iterrows():
        print(f"  {i+1}. {row['fai_name']}")
        print(f"     投资增速: {row['fai_yoy']:+.2f}%")
    
    print(f"\n配置建议:")
    if latest_summary['avg_fai_yoy'] > 10:
        print(f"  ✓ 建议超配AI相关行业")
        print(f"  ✓ 关注芯片、服务器、数据中心基础设施")
        print(f"  ✓ 重点布局算力相关标的")
    elif latest_summary['avg_fai_yoy'] > 5:
        print(f"  ✓ 建议标配AI相关行业")
        print(f"  ✓ 择机布局优质龙头")
    else:
        print(f"  ✗ 建议低配或观望")
        print(f"  ✗ 等待投资增速回升信号")


def main():
    """主函数"""
    analyze_ai_investment()


if __name__ == '__main__':
    main()