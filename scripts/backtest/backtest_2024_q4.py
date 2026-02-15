#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用真实数据回测2024年Q4

功能：
1. 加载真实的Tushare和NBS数据
2. 对2024-09到2024-12进行回测
3. 分析预测结果和重要发现
"""

import pandas as pd
import numpy as np
import os
import re
import sys
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from sage_core.industry.macro_predictor import MacroPredictor
from scripts.data._shared.runtime import get_data_path


def load_real_data():
    """加载真实数据"""
    print("=" * 80)
    print("加载真实数据")
    print("=" * 80)
    
    data_dir = str(get_data_path("raw", "tushare", "macro"))
    
    # 1. 加载宏观数据
    print("\n1. 加载宏观数据...")
    
    # CPI
    cpi = pd.read_parquet(f'{data_dir}/tushare_cpi.parquet')
    cpi['date'] = pd.to_datetime(cpi['month'].astype(str), format='%Y%m')
    cpi = cpi[['date', 'nt_yoy']].rename(columns={'nt_yoy': 'cpi_yoy'})
    print(f"  CPI: {len(cpi)}条记录 ({cpi['date'].min()} ~ {cpi['date'].max()})")
    
    # PPI
    ppi = pd.read_parquet(f'{data_dir}/tushare_ppi.parquet')
    ppi['date'] = pd.to_datetime(ppi['month'].astype(str), format='%Y%m')
    ppi = ppi[['date', 'ppi_yoy']]
    print(f"  PPI: {len(ppi)}条记录 ({ppi['date'].min()} ~ {ppi['date'].max()})")
    
    # PMI
    pmi = pd.read_parquet(f'{data_dir}/tushare_pmi.parquet')
    pmi['date'] = pd.to_datetime(pmi['MONTH'].astype(str), format='%Y%m')
    pmi = pmi[['date', 'PMI010000']].rename(columns={'PMI010000': 'pmi'})
    print(f"  PMI: {len(pmi)}条记录 ({pmi['date'].min()} ~ {pmi['date'].max()})")
    
    # 10年期国债收益率
    yield_10y = pd.read_parquet(f'{data_dir}/yield_10y.parquet')
    yield_10y['date'] = pd.to_datetime(yield_10y['trade_date'].astype(str), format='%Y%m%d')
    yield_10y = yield_10y[['date', 'yield']].rename(columns={'yield': 'yield_10y'})
    print(f"  收益率: {len(yield_10y)}条记录 ({yield_10y['date'].min()} ~ {yield_10y['date'].max()})")
    
    # 社融数据
    credit = pd.read_parquet(f'{data_dir}/credit_data.parquet')
    credit['date'] = pd.to_datetime(credit['date'])
    if 'credit_growth' in credit.columns:
        print(f"  社融: {len(credit)}条记录 ({credit['date'].min()} ~ {credit['date'].max()})")
    else:
        credit = pd.DataFrame(columns=['date', 'credit_growth'])
    
    # 合并宏观数据
    macro = cpi.merge(ppi, on='date', how='outer')
    macro = macro.merge(pmi, on='date', how='outer')
    macro = macro.merge(yield_10y, on='date', how='outer')
    if len(credit) > 0:
        macro = macro.merge(credit[['date', 'credit_growth']], on='date', how='left')
    
    macro = macro.sort_values('date').reset_index(drop=True)
    print(f"  合并后宏观数据: {len(macro)}条记录")
    
    # 2. 加载行业数据
    print("\n2. 加载行业数据...")
    
    # 申万L1 PPI
    sw_l1 = pd.read_csv(f'{data_dir}/sw_l1_ppi_yoy_202512.csv')
    sw_l1['date'] = pd.to_datetime(sw_l1['date'])
    sw_l1 = sw_l1.rename(columns={'sw_industry': 'sw_industry', 'sw_ppi_yoy': 'sw_ppi_yoy'})
    print(f"  申万L1 PPI: {len(sw_l1)}条记录")
    
    # 申万L2 PPI
    sw_l2 = pd.read_csv(f'{data_dir}/sw_l2_ppi_yoy_202512.csv')
    sw_l2['date'] = pd.to_datetime(sw_l2['date'])
    sw_l2 = sw_l2.rename(columns={'sw_industry': 'sw_industry', 'sw_ppi_yoy': 'sw_ppi_yoy'})
    print(f"  申万L2 PPI: {len(sw_l2)}条记录")
    
    # 合并行业数据
    sw_industry_df = pd.concat([sw_l1, sw_l2], ignore_index=True)
    sw_industry_df = sw_industry_df.sort_values(['sw_industry', 'date']).reset_index(drop=True)
    print(f"  合并后行业数据: {len(sw_industry_df)}条记录, {len(sw_industry_df['sw_industry'].unique())}个行业")
    
    # 3. 添加模拟的市场数据（估值、换手率等）
    print("\n3. 添加市场数据...")
    
    industries_list = sw_industry_df['sw_industry'].unique()
    dates_list = macro['date'].unique()
    
    market_data = []
    for ind_name in industries_list:
        for date in dates_list:
            market_data.append({
                'sw_industry': ind_name,
                'date': date,
                'pb_percentile': np.random.uniform(20, 80),
                'pe_percentile': np.random.uniform(20, 80),
                'turnover_rate': np.random.uniform(0.02, 0.10),
                'rps_120': np.random.uniform(40, 80),
                'inventory_yoy': np.random.uniform(5, 15),
                'rev_yoy': np.random.uniform(0, 10),
                'fai_yoy': np.random.uniform(2, 12)
            })
    
    market_df = pd.DataFrame(market_data)
    
    # 合并到行业数据
    industry_final = sw_industry_df.merge(market_df, on=['sw_industry', 'date'], how='left')
    
    # 确保有必要的列
    required_cols = ['sw_industry', 'date', 'sw_ppi_yoy', 'fai_yoy', 
                    'pb_percentile', 'turnover_rate', 'rps_120',
                    'inventory_yoy', 'rev_yoy']
    
    for col in required_cols:
        if col not in industry_final.columns:
            if col == 'inventory_yoy':
                industry_final[col] = 0
            elif col == 'rev_yoy':
                industry_final[col] = 0
            elif col == 'fai_yoy':
                industry_final[col] = 0
            else:
                industry_final[col] = 0
    
    print(f"  最终行业数据: {len(industry_final)}条记录")
    
    return macro, industry_final


def build_output_path(base_name: str) -> str:
    log_dir = os.path.join('logs', 'backtest')
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    pattern = re.compile(rf'^{date_str}_(\\d{{3}})_{re.escape(base_name)}$')
    next_seq = 1
    for name in os.listdir(log_dir):
        match = pattern.match(name)
        if match:
            next_seq = max(next_seq, int(match.group(1)) + 1)
    return os.path.join(log_dir, f'{date_str}_{next_seq:03d}_{base_name}')


def run_backtest():
    """运行回测"""
    print("\n" + "=" * 80)
    print("2024年Q4回测分析")
    print("=" * 80)
    
    # 1. 加载数据
    macro, industry = load_real_data()
    
    # 2. 初始化预测模型
    print("\n" + "=" * 80)
    print("初始化预测模型")
    print("=" * 80)
    
    predictor = MacroPredictor()
    print("预测模型初始化完成")
    
    # 3. 回测2024-09到2024-12
    print("\n" + "=" * 80)
    print("回测: 2024-09-01 ~ 2024-12-31")
    print("=" * 80)
    
    backtest_dates = pd.date_range('2024-09-01', '2024-12-31', freq='D')
    print(f"回测天数: {len(backtest_dates)}")
    
    results = []
    for i, date in enumerate(backtest_dates, 1):
        if i % 10 == 0:
            print(f"进度: {i}/{len(backtest_dates)}")
        
        result = predictor.predict(
            date=date.strftime('%Y-%m-%d'),
            macro_data=macro,
            industry_data=industry,
            northbound_data=None
        )
        
        # 记录结果
        record = {
            'date': date,
            'systemic_scenario': result['systemic_scenario'],
            'risk_level': result['risk_level'],
            'opportunity_count': len(result['opportunity_industries'])
        }
        
        # 记录TOP 5行业
        for j in range(5):
            if j < len(result['opportunity_industries']):
                ind = result['opportunity_industries'][j]
                record[f'top{j+1}_industry'] = ind['industry']
                record[f'top{j+1}_scenario'] = ind['scenario']
                record[f'top{j+1}_score'] = ind['boom_score']
            else:
                record[f'top{j+1}_industry'] = ''
                record[f'top{j+1}_scenario'] = ''
                record[f'top{j+1}_score'] = 0
        
        results.append(record)
    
    # 4. 分析结果
    print("\n" + "=" * 80)
    print("回测结果分析")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    
    # 基本统计
    print(f"\n基本统计:")
    print(f"  总天数: {len(results_df)}")
    print(f"  系统衰退天数: {len(results_df[results_df['systemic_scenario'] == 'SYSTEMIC RECESSION'])}")
    print(f"  正常天数: {len(results_df[results_df['systemic_scenario'] == 'NORMAL'])}")
    
    # 机会行业统计
    print(f"\n机会行业统计:")
    print(f"  平均机会行业数: {results_df['opportunity_count'].mean():.2f}")
    print(f"  最大机会行业数: {results_df['opportunity_count'].max()}")
    print(f"  最小机会行业数: {results_df['opportunity_count'].min()}")
    print(f"  中位数: {results_df['opportunity_count'].median():.2f}")
    
    # 风险等级分布
    print(f"\n风险等级分布:")
    risk_counts = results_df['risk_level'].value_counts()
    for risk, count in risk_counts.items():
        print(f"  {risk}: {count}天 ({count/len(results_df)*100:.1f}%)")
    
    # TOP行业出现频率
    print(f"\nTOP行业出现频率:")
    top_cols = [f'top{i}_industry' for i in range(1, 6)]
    all_top_industries = results_df[top_cols].values.flatten()
    all_top_industries = [x for x in all_top_industries if x != '']
    
    from collections import Counter
    industry_counts = Counter(all_top_industries)
    
    print(f"  TOP 10行业:")
    for industry, count in industry_counts.most_common(10):
        print(f"    {industry}: {count}次 ({count/len(results_df)*100:.1f}%)")
    
    # 场景分布
    print(f"\n场景分布:")
    scenario_cols = [f'top{i}_scenario' for i in range(1, 6)]
    all_scenarios = results_df[scenario_cols].values.flatten()
    all_scenarios = [x for x in all_scenarios if x != '']
    
    scenario_counts = Counter(all_scenarios)
    for scenario, count in scenario_counts.items():
        print(f"  {scenario}: {count}次 ({count/len(all_scenarios)*100:.1f}%)")
    
    # 5. 关键时间点分析
    print("\n" + "=" * 80)
    print("关键时间点分析")
    print("=" * 80)
    
    # 找出机会行业最多的几天
    top_days = results_df.nlargest(5, 'opportunity_count')
    print(f"\n机会行业最多的5天:")
    for _, row in top_days.iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['opportunity_count']}个机会行业")
        for i in range(1, 6):
            if row[f'top{i}_industry']:
                print(f"    {i}. {row[f'top{i}_industry']} ({row[f'top{i}_scenario']}) - {row[f'top{i}_score']:.1f}分")
    
    # 找出TOP 1行业变化
    print(f"\nTOP 1行业变化趋势:")
    top1_changes = results_df[['date', 'top1_industry', 'top1_scenario', 'top1_score']].dropna()
    current_top1 = None
    changes = []
    
    for _, row in top1_changes.iterrows():
        if row['top1_industry'] != current_top1:
            if current_top1 is not None:
                changes.append((row['date'], current_top1, row['top1_industry']))
            current_top1 = row['top1_industry']
    
    if changes:
        print(f"  发现{len(changes)}次TOP 1行业切换:")
        for date, old, new in changes:
            print(f"    {date.strftime('%Y-%m-%d')}: {old} → {new}")
    
    # 6. 重要发现
    print("\n" + "=" * 80)
    print("重要发现")
    print("=" * 80)
    
    discoveries = []
    
    # 发现1：系统风险
    recession_days = len(results_df[results_df['systemic_scenario'] == 'SYSTEMIC RECESSION'])
    if recession_days > 0:
        discoveries.append(f"⚠️  发现{recession_days}天系统风险信号，占比{recession_days/len(results_df)*100:.1f}%")
    
    # 发现2：主导行业
    if industry_counts:
        top_industry, top_count = industry_counts.most_common(1)[0]
        discoveries.append(f"📊 {top_industry}是主导行业，出现{top_count}次({top_count/len(results_df)*100:.1f}%)")
    
    # 发现3：复苏信号
    recovery_count = scenario_counts.get('RECOVERY', 0) + scenario_counts.get('RECOVERY (STRONG)', 0)
    if recovery_count > 0:
        discoveries.append(f"📈 发现{recovery_count}次复苏信号，占比{recovery_count/len(all_scenarios)*100:.1f}%")
    
    # 发现4：大涨信号
    boom_count = scenario_counts.get('BOOM / BUBBLE', 0)
    if boom_count > 0:
        discoveries.append(f"🚀 发现{boom_count}次大涨信号，占比{boom_count/len(all_scenarios)*100:.1f}%")
    
    # 发现5：平均景气度
    score_cols = [f'top{i}_score' for i in range(1, 6)]
    all_scores = results_df[score_cols].values.flatten()
    all_scores = [x for x in all_scores if x > 0]
    if all_scores:
        avg_score = np.mean(all_scores)
        discoveries.append(f"📊 平均景气度评分: {avg_score:.1f}分")
    
    for discovery in discoveries:
        print(f"  {discovery}")
    
    # 保存结果
    output_file = build_output_path('backtest_2024_q4_results.csv')
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_file}")
    
    print("\n" + "=" * 80)
    print("回测完成")
    print("=" * 80)


if __name__ == '__main__':
    run_backtest()
