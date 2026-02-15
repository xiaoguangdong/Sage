#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
宏观经济预测系统测试脚本

功能：
1. 测试数据加载
2. 测试预测模型
3. 测试完整预测流程
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from sage_core.industry.macro_predictor import MacroPredictor


def create_test_data():
    """创建测试数据"""
    print("=" * 80)
    print("创建测试数据")
    print("=" * 80)
    
    # 创建日期范围（2020-2024）
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='ME')
    
    # 1. 创建宏观数据
    macro_data = pd.DataFrame({
        'date': dates,
        'credit_growth': np.random.uniform(9, 14, len(dates)),
        'pmi': np.random.uniform(48, 52, len(dates)),
        'cpi_yoy': np.random.uniform(0, 3, len(dates)),
        'yield_10y': np.random.uniform(2.5, 3.5, len(dates))
    })
    
    # 模拟2020年初疫情导致的PMI下降
    macro_data.loc[macro_data['date'] < pd.Timestamp('2020-06-01'), 'pmi'] = np.random.uniform(35, 48, len(macro_data[macro_data['date'] < pd.Timestamp('2020-06-01')]))
    
    # 模拟2024年经济复苏
    macro_data.loc[macro_data['date'] >= pd.Timestamp('2024-09-01'), 'pmi'] = np.random.uniform(50, 52, len(macro_data[macro_data['date'] >= pd.Timestamp('2024-09-01')]))
    macro_data.loc[macro_data['date'] >= pd.Timestamp('2024-09-01'), 'credit_growth'] = np.random.uniform(11, 14, len(macro_data[macro_data['date'] >= pd.Timestamp('2024-09-01')]))
    
    print(f"宏观数据: {len(macro_data)}条记录")
    print(f"日期范围: {macro_data['date'].min()} ~ {macro_data['date'].max()}")
    
    # 2. 创建行业数据
    industries = ['电子', '汽车', '医药', '食品饮料', '基础化工', '钢铁', '有色金属', 
                  '电力设备', '家用电器', '纺织服饰', '轻工制造', '农林牧渔']
    
    industry_rows = []
    for industry in industries:
        for date in dates:
            # 模拟不同行业的景气度变化
            if industry == '电子':
                # 电子行业：2020年低迷，2024年复苏
                if date < pd.Timestamp('2020-06-01'):
                    ppi_yoy = np.random.uniform(-8, -5)
                    fai_yoy = np.random.uniform(-5, 0)
                    pb_percentile = np.random.uniform(10, 30)
                elif date >= pd.Timestamp('2024-09-01'):
                    ppi_yoy = np.random.uniform(-1, 3)
                    fai_yoy = np.random.uniform(8, 15)
                    pb_percentile = np.random.uniform(40, 60)
                else:
                    ppi_yoy = np.random.uniform(-5, 2)
                    fai_yoy = np.random.uniform(0, 8)
                    pb_percentile = np.random.uniform(20, 50)
            elif industry == '汽车':
                # 汽车行业：2024年大涨
                if date >= pd.Timestamp('2024-09-01'):
                    ppi_yoy = np.random.uniform(2, 5)
                    fai_yoy = np.random.uniform(15, 25)
                    pb_percentile = np.random.uniform(70, 90)
                    turnover_rate = np.random.uniform(0.10, 0.15)
                    rps_120 = np.random.uniform(85, 98)
                else:
                    ppi_yoy = np.random.uniform(-3, 3)
                    fai_yoy = np.random.uniform(0, 10)
                    pb_percentile = np.random.uniform(20, 60)
                    turnover_rate = np.random.uniform(0.02, 0.08)
                    rps_120 = np.random.uniform(40, 70)
            else:
                # 其他行业：正常波动
                ppi_yoy = np.random.uniform(-5, 5)
                fai_yoy = np.random.uniform(0, 10)
                pb_percentile = np.random.uniform(20, 80)
                turnover_rate = np.random.uniform(0.02, 0.10)
                rps_120 = np.random.uniform(30, 80)
            
            industry_rows.append({
                'sw_industry': industry,
                'date': date,
                'sw_ppi_yoy': ppi_yoy,  # 修改为sw_ppi_yoy以匹配代码期望
                'fai_yoy': fai_yoy,
                'inventory_yoy': np.random.uniform(0, 15),
                'rev_yoy': np.random.uniform(-5, 15),
                'pb_percentile': pb_percentile,
                'pe_percentile': pb_percentile + np.random.uniform(-10, 10),
                'turnover_rate': turnover_rate if 'turnover_rate' in locals() else np.random.uniform(0.02, 0.10),
                'rps_120': rps_120 if 'rps_120' in locals() else np.random.uniform(30, 80)
            })
    
    industry_data = pd.DataFrame(industry_rows)
    
    print(f"行业数据: {len(industry_data)}条记录")
    print(f"行业数量: {len(industries)}")
    
    return macro_data, industry_data


def test_prediction():
    """测试预测系统"""
    print("\n" + "=" * 80)
    print("宏观经济预测系统测试")
    print("=" * 80)
    print()
    
    # 1. 创建测试数据
    macro_data, industry_data = create_test_data()
    
    # 2. 初始化预测模型
    print("\n" + "=" * 80)
    print("初始化预测模型")
    print("=" * 80)
    
    predictor = MacroPredictor()
    print("预测模型初始化完成")
    
    # 3. 测试关键时间点的预测
    test_dates = [
        ('2020-02-01', '疫情初期'),
        ('2020-07-01', '疫情恢复期'),
        ('2023-12-01', '正常期'),
        ('2024-09-15', '牛市启动'),
        ('2024-11-01', '牛市中期'),
        ('2024-12-01', '牛市后期')
    ]
    
    for date, description in test_dates:
        print("\n" + "=" * 80)
        print(f"测试时间点: {date} ({description})")
        print("=" * 80)
        
        result = predictor.predict(
            date=date,
            macro_data=macro_data,
            industry_data=industry_data,
            northbound_data=None
        )
        
        print(f"\n系统场景: {result['systemic_scenario']}")
        print(f"风险等级: {result['risk_level']}")
        print(f"摘要: {result['summary']}")
        
        if result['opportunity_industries']:
            print(f"\n机会行业 ({len(result['opportunity_industries'])}个):")
            for i, ind in enumerate(result['opportunity_industries'][:5], 1):
                print(f"  {i}. {ind['industry']} ({ind['scenario']}) - 景气度: {ind['boom_score']:.1f}")
                print(f"     PPI: {ind['key_indicators']['ppi_yoy']:.2f}%, "
                      f"FAI: {ind['key_indicators']['fai_yoy']:.2f}%, "
                      f"PB分位: {ind['key_indicators']['pb_percentile']:.1f}%")
        else:
            print("\n暂无机会行业")
    
    # 4. 回测2024-09到2024-12
    print("\n" + "=" * 80)
    print("回测: 2024-09-01 ~ 2024-12-31")
    print("=" * 80)
    
    backtest_dates = pd.date_range('2024-09-01', '2024-12-31', freq='D')
    
    print(f"回测天数: {len(backtest_dates)}")
    
    results = []
    for i, date in enumerate(backtest_dates, 1):
        if i % 10 == 0:  # 每10天输出一次进度
            print(f"进度: {i}/{len(backtest_dates)}")
        
        result = predictor.predict(
            date=date.strftime('%Y-%m-%d'),
            macro_data=macro_data,
            industry_data=industry_data
        )
        
        results.append({
            'date': date,
            'systemic_scenario': result['systemic_scenario'],
            'opportunity_count': len(result['opportunity_industries']),
            'top_industry': result['opportunity_industries'][0]['industry'] if result['opportunity_industries'] else None
        })
    
    # 统计结果
    results_df = pd.DataFrame(results)
    
    print("\n回测结果统计:")
    print(f"总天数: {len(results_df)}")
    print(f"系统衰退天数: {len(results_df[results_df['systemic_scenario'] == 'SYSTEMIC RECESSION'])}")
    print(f"平均机会行业数: {results_df['opportunity_count'].mean():.2f}")
    print(f"最大机会行业数: {results_df['opportunity_count'].max()}")
    
    # 统计TOP行业出现次数
    top_industry_counts = results_df['top_industry'].value_counts()
    print(f"\nTOP行业出现次数:")
    for industry, count in top_industry_counts.head(5).items():
        print(f"  {industry}: {count}次 ({count/len(results_df)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == '__main__':
    test_prediction()
