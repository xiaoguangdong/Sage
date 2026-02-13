#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用真实宏观数据+合理模拟行业数据进行2024年Q4回测

功能：
1. 使用真实的CPI、PPI、PMI、收益率数据
2. 模拟2024年Q4的行业景气度变化（参考实际市场情况）
3. 分析预测结果和重要发现
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from ml_stock_forecast.models.macro_predictor import MacroPredictor


def load_realistic_data():
    """加载真实宏观数据+模拟行业数据"""
    print("=" * 80)
    print("加载数据（真实宏观+合理模拟行业）")
    print("=" * 80)
    
    data_dir = 'data/tushare/macro'
    
    # 1. 加载真实宏观数据
    print("\n1. 加载真实宏观数据...")
    
    # CPI
    cpi = pd.read_parquet(f'{data_dir}/tushare_cpi.parquet')
    cpi['date'] = pd.to_datetime(cpi['month'].astype(str), format='%Y%m')
    cpi = cpi[['date', 'nt_yoy']].rename(columns={'nt_yoy': 'cpi_yoy'})
    print(f"  CPI: {len(cpi)}条记录")
    
    # PPI
    ppi = pd.read_parquet(f'{data_dir}/tushare_ppi.parquet')
    ppi['date'] = pd.to_datetime(ppi['month'].astype(str), format='%Y%m')
    ppi = ppi[['date', 'ppi_yoy']]
    print(f"  PPI: {len(ppi)}条记录")
    
    # PMI
    pmi = pd.read_parquet(f'{data_dir}/tushare_pmi.parquet')
    pmi['date'] = pd.to_datetime(pmi['MONTH'].astype(str), format='%Y%m')
    pmi = pmi[['date', 'PMI010000']].rename(columns={'PMI010000': 'pmi'})
    print(f"  PMI: {len(pmi)}条记录")
    
    # 10年期国债收益率
    yield_10y = pd.read_parquet(f'{data_dir}/yield_10y.parquet')
    yield_10y['date'] = pd.to_datetime(yield_10y['trade_date'].astype(str), format='%Y%m%d')
    yield_10y = yield_10y[['date', 'yield']].rename(columns={'yield': 'yield_10y'})
    print(f"  收益率: {len(yield_10y)}条记录")
    
    # 合并宏观数据
    macro = cpi.merge(ppi, on='date', how='outer')
    macro = macro.merge(pmi, on='date', how='outer')
    macro = macro.merge(yield_10y, on='date', how='outer')
    
    # 添加社融增速（模拟）
    macro['credit_growth'] = np.random.uniform(9.5, 11.5, len(macro))
    
    macro = macro.sort_values('date').reset_index(drop=True)
    print(f"  合并后宏观数据: {len(macro)}条记录")
    
    # 2. 加载北向资金数据
    print("\n2. 加载北向资金数据...")
    
    # 加载北向资金数据
    northbound_flow = pd.read_parquet('data/tushare/northbound/daily_flow.parquet')
    northbound_flow['trade_date'] = pd.to_datetime(northbound_flow['trade_date'].astype(str), format='%Y%m%d')
    northbound_flow = northbound_flow.sort_values('trade_date')
    
    # 转换为数值类型
    northbound_flow['ggt_ss'] = pd.to_numeric(northbound_flow['ggt_ss'], errors='coerce')  # 沪股通（上交所北向累计持仓市值）
    northbound_flow['ggt_sz'] = pd.to_numeric(northbound_flow['ggt_sz'], errors='coerce')  # 深股通（深交所北向累计持仓市值）
    
    # 计算日度净流入额（当日累计市值 - 前一日累计市值）
    northbound_flow['north_inflow'] = northbound_flow['ggt_ss'].diff()  # 沪股通日度净流入
    northbound_flow['south_inflow'] = northbound_flow['ggt_sz'].diff()  # 深股通日度净流入
    # 北向资金日度净流入 = 沪股通日度净流入 + 深股通日度净流入
    northbound_flow['net_flow'] = northbound_flow['north_inflow'] + northbound_flow['south_inflow']
    
    print(f"  ⚠️  注意: 北向资金数据说明:")
    print(f"     - ggt_ss 和 ggt_sz 是累计持仓市值（亿元）")
    print(f"     - 通过差分计算日度净流入额（net_flow）")
    print(f"     - 2024年Q4平均日度净流入: {northbound_flow[northbound_flow['trade_date'].between('2024-09-01', '2024-12-31')]['net_flow'].mean():.2f}亿元")
    
    # 按行业汇总北向资金流向（需要根据代码映射到行业，这里简化处理）
    # 创建模拟的行业北向资金数据
    industries = [
        '非银金融', '银行', '房地产', '建筑装饰', '建筑材料',
        '电子', '计算机', '通信', '传媒',
        '汽车', '电力设备', '家用电器',
        '食品饮料', '农林牧渔', '商贸零售', '社会服务',
        '医药生物', '基础化工', '有色金属', '钢铁', '煤炭',
        '石油石化', '交通运输', '公用事业'
    ]
    
    dates = macro['date'].unique()
    northbound_industry = []
    
    for industry in industries:
        for date in dates:
            # 模拟行业北向资金数据
            base_flow = np.random.uniform(-50000, 150000)
            if date >= pd.Timestamp('2024-09-01'):
                # 政策受益板块北向资金流入更多
                if industry in ['非银金融', '银行', '房地产']:
                    base_flow = np.random.uniform(50000, 200000)
                elif industry in ['电子', '计算机', '通信']:
                    base_flow = np.random.uniform(30000, 180000)
                elif industry in ['汽车', '电力设备']:
                    base_flow = np.random.uniform(20000, 150000)
            
            northbound_industry.append({
                'industry_name': industry,
                'trade_date': date,
                'north_money': max(0, base_flow),
                'south_money': max(0, -base_flow),
                'net_flow': base_flow,  # 添加净流入字段
                'northbound_signal': 1 if base_flow > 50000 else 0,
                'industry_ratio': np.random.uniform(0.01, 0.08)
            })
    
    northbound_industry_df = pd.DataFrame(northbound_industry)
    northbound_industry_df = northbound_industry_df.sort_values(['industry_name', 'trade_date']).reset_index(drop=True)
    print(f"  北向资金行业数据: {len(northbound_industry_df)}条记录")
    
    # 3. 加载申万行业数据（基于NBS数据）
    print("\n3. 加载申万行业数据（基于NBS数据）...")

    # 加载申万-NBS映射后的行业数据
    nbs_result = load_nbs_industry_data(start_date='2020-01-01', end_date='2026-12-31')

    # 提取PPI和FAI数据
    sw_ppi = nbs_result['industry_ppi']
    sw_fai = nbs_result['industry_fai']

    # 合并PPI和FAI数据
    industry_df = sw_ppi.merge(sw_fai, on=['sw_industry', 'date'], how='outer')

    # 只保留需要的日期
    industry_df = industry_df[industry_df['date'].isin(dates)]

    # 重命名列以匹配模型期望的字段名
    industry_df = industry_df.rename(columns={'ppi_yoy': 'sw_ppi_yoy'})

    # 添加模拟的估值和流动性数据（因为没有真实的估值数据）
    # 这些数据仍然使用模拟，因为需要从股票市场获取
    industry_df['pb_percentile'] = np.random.uniform(20, 80, len(industry_df))
    industry_df['pe_percentile'] = industry_df['pb_percentile'] + np.random.uniform(-10, 10, len(industry_df))
    industry_df['turnover_rate'] = np.random.uniform(0.02, 0.12, len(industry_df))
    industry_df['rps_120'] = np.random.uniform(40, 80, len(industry_df))
    industry_df['inventory_yoy'] = np.random.uniform(5, 15, len(industry_df))
    industry_df['rev_yoy'] = np.random.uniform(0, 10, len(industry_df))

    industry_df = industry_df.sort_values(['sw_industry', 'date']).reset_index(drop=True)
    print(f"  行业数据: {len(industry_df)}条记录, {len(industry_df['sw_industry'].unique())}个行业")
    print(f"  ⚠️  说明: PPI和FAI数据来自NBS真实数据")
    print(f"  ⚠️  估值和流动性数据使用模拟（需要从股票市场获取）")

    return macro, industry_df, northbound_industry_df


def load_nbs_industry_data(start_date='2020-01-01', end_date='2026-12-31'):
    """
    加载NBS数据并映射到申万行业

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        dict: 包含申万行业数据的字典
    """
    import yaml

    data_dir = 'data/tushare/macro'

    # 1. 读取NBS PPI数据
    ppi_data = pd.read_csv(f'{data_dir}/nbs_ppi_industry_2020.csv')
    ppi_data['date'] = pd.to_datetime(ppi_data['date'].astype(str), format='%Y-%m-%d')

    # 清理行业名称（移除后缀）
    ppi_data['industry_clean'] = ppi_data['industry'].str.replace('工业生产者出厂价格指数(上月=100)', '')

    # 2. 读取NBS FAI数据
    fai_data = pd.read_csv(f'{data_dir}/nbs_fai_industry_2020.csv')
    fai_data['date'] = pd.to_datetime(fai_data['date'].astype(str), format='%Y-%m-%d')

    # 清理行业名称
    fai_data['industry_clean'] = fai_data['industry'].str.replace('固定资产投资额累计同比增长率(%)', '')

    # 3. 读取映射配置
    config_path = 'config/sw_nbs_mapping.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    sw_to_nbs = config['sw_to_nbs']

    # 4. 将NBS数据映射到申万行业
    sw_industries = list(sw_to_nbs.keys())
    dates = pd.date_range(start_date, end_date, freq='ME')

    # 创建申万行业PPI数据
    industry_ppi = []
    for sw_industry in sw_industries:
        nbs_mappings = sw_to_nbs[sw_industry]

        for date in dates:
            # 计算该申万行业在该日期的PPI（基于权重聚合）
            total_weight = 0
            weighted_ppi = 0

            for mapping in nbs_mappings:
                nbs_industry = mapping['nbs_industry']
                weight = mapping['weight']

                # 在PPI数据中查找该NBS行业的数据
                ppi_record = ppi_data[
                    (ppi_data['industry_clean'].str.contains(nbs_industry, na=False)) &
                    (ppi_data['date'].dt.year == date.year) &
                    (ppi_data['date'].dt.month == date.month)
                ]

                if len(ppi_record) > 0:
                    # 使用环比数据
                    ppi_mom = ppi_record['ppi_mom'].mean()
                    weighted_ppi += ppi_mom * weight
                    total_weight += weight

            # 计算加权平均PPI
            if total_weight > 0:
                ppi_value = weighted_ppi / total_weight
            else:
                ppi_value = 100.0  # 默认值

            industry_ppi.append({
                'sw_industry': sw_industry,
                'date': date,
                'ppi_mom': ppi_value
            })

    industry_ppi_df = pd.DataFrame(industry_ppi)

    # 5. 将FAI数据映射到申万行业
    industry_fai = []
    for sw_industry in sw_industries:
        nbs_mappings = sw_to_nbs[sw_industry]

        for date in dates:
            # 计算该申万行业在该日期的FAI（基于权重聚合）
            total_weight = 0
            weighted_fai = 0

            for mapping in nbs_mappings:
                nbs_industry = mapping['nbs_industry']
                weight = mapping['weight']

                # 在FAI数据中查找该NBS行业的数据
                fai_record = fai_data[
                    (fai_data['industry_clean'].str.contains(nbs_industry, na=False)) &
                    (fai_data['date'].dt.year == date.year) &
                    (fai_data['date'].dt.month == date.month)
                ]

                if len(fai_record) > 0:
                    fai_value = fai_record['fai_yoy'].mean()
                    weighted_fai += fai_value * weight
                    total_weight += weight

            # 计算加权平均FAI
            if total_weight > 0:
                fai_value = weighted_fai / total_weight
            else:
                fai_value = 0.0  # 默认值

            industry_fai.append({
                'sw_industry': sw_industry,
                'date': date,
                'fai_yoy': fai_value
            })

    industry_fai_df = pd.DataFrame(industry_fai)

    # 6. 计算PPI同比数据
    industry_ppi_df = industry_ppi_df.sort_values(['sw_industry', 'date'])

    for sw_industry in sw_industries:
        sw_data = industry_ppi_df[industry_ppi_df['sw_industry'] == sw_industry].copy()

        # 计算累计指数
        sw_data['cumulative_ppi'] = 100.0
        for i in range(1, len(sw_data)):
            sw_data.iloc[i, sw_data.columns.get_loc('cumulative_ppi')] = (
                sw_data.iloc[i-1]['cumulative_ppi'] * sw_data.iloc[i]['ppi_mom'] / 100
            )

        # 计算同比（与去年同期相比）
        sw_data['ppi_yoy'] = 0.0
        for i in range(12, len(sw_data)):
            sw_data.iloc[i, sw_data.columns.get_loc('ppi_yoy')] = (
                (sw_data.iloc[i]['cumulative_ppi'] / sw_data.iloc[i-12]['cumulative_ppi'] - 1) * 100
            )

        # 更新数据
        industry_ppi_df.loc[industry_ppi_df['sw_industry'] == sw_industry, 'cumulative_ppi'] = sw_data['cumulative_ppi'].values
        industry_ppi_df.loc[industry_ppi_df['sw_industry'] == sw_industry, 'ppi_yoy'] = sw_data['ppi_yoy'].values

    # 7. 返回结果
    result = {
        'industry_ppi': industry_ppi_df,
        'industry_fai': industry_fai_df,
        'sw_industries': sw_industries
    }

    return result


def run_backtest():
    """运行回测"""
    print("\n" + "=" * 80)
    print("2024年Q4回测分析（真实宏观+合理模拟）")
    print("=" * 80)
    
    # 1. 加载数据
    macro, industry, northbound = load_realistic_data()
    
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
            northbound_data=northbound  # 传入北向资金数据
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
    else:
        discoveries.append(f"✅ 无系统风险信号，市场环境相对稳定")
    
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
        max_score = np.max(all_scores)
        discoveries.append(f"📊 平均景气度评分: {avg_score:.1f}分，最高: {max_score:.1f}分")
    
    # 发现6：行业轮动
    if len(changes) > 5:
        discoveries.append(f"🔄 行业轮动频繁，切换{len(changes)}次，说明市场结构快速变化")
    elif len(changes) > 0:
        discoveries.append(f"🔄 行业轮动适中，切换{len(changes)}次")
    
    # 发现7：宏观环境
    avg_pmi = macro[macro['date'] >= pd.Timestamp('2024-09-01')]['pmi'].mean()
    avg_ppi = macro[macro['date'] >= pd.Timestamp('2024-09-01')]['ppi_yoy'].mean()
    avg_cpi = macro[macro['date'] >= pd.Timestamp('2024-09-01')]['cpi_yoy'].mean()
    avg_yield = macro[macro['date'] >= pd.Timestamp('2024-09-01')]['yield_10y'].mean()
    discoveries.append(f"🌡️  2024年Q4宏观环境: PMI平均{avg_pmi:.1f}, PPI平均{avg_ppi:.2f}%, CPI平均{avg_cpi:.2f}%, 10Y国债{avg_yield:.2f}%")
    
    # 发现8：北向资金
    avg_net_flow = northbound[northbound['trade_date'] >= pd.Timestamp('2024-09-01')]['net_flow'].mean()
    discoveries.append(f"💰 北向资金平均净流入: {avg_net_flow/10000:.1f}亿元")
    
    for discovery in discoveries:
        print(f"  {discovery}")
    
    # 保存结果
    output_file = 'backtest_2024_q4_realistic_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_file}")
    
    print("\n" + "=" * 80)
    print("回测完成")
    print("=" * 80)


if __name__ == '__main__':
    run_backtest()
