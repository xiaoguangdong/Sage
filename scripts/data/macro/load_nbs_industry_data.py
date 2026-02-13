#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
加载NBS数据并映射到申万行业

功能：
1. 读取NBS PPI和FAI数据
2. 根据映射配置将NBS数据映射到申万行业
3. 输出申万行业的PPI和FAI数据
"""

import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime

def load_nbs_industry_data(start_date='2020-01-01', end_date='2026-12-31'):
    """
    加载NBS数据并映射到申万行业

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        dict: 包含申万行业数据的字典
            - 'industry_ppi': 申万行业PPI数据
            - 'industry_fai': 申万行业FAI数据
            - 'sw_industries': 申万行业列表
    """
    data_dir = 'data/tushare/macro'

    print("=" * 80)
    print("加载NBS数据并映射到申万行业")
    print("=" * 80)

    # 1. 读取NBS PPI数据
    print("\n1. 读取NBS PPI数据...")
    ppi_data = pd.read_csv(f'{data_dir}/nbs_ppi_industry_2020.csv')
    ppi_data['date'] = pd.to_datetime(ppi_data['date'].astype(str), format='%Y-%m-%d')
    print(f"  原始数据: {len(ppi_data)}条记录")

    # 清理行业名称（移除后缀）
    ppi_data['industry_clean'] = ppi_data['industry'].str.replace('工业生产者出厂价格指数(上月=100)', '')
    ppi_data['industry_clean'] = ppi_data['industry_clean'].str.replace('固定资产投资额累计同比增长率(%)', '')

    # 2. 读取NBS FAI数据
    print("\n2. 读取NBS FAI数据...")
    fai_data = pd.read_csv(f'{data_dir}/nbs_fai_industry_2020.csv')
    fai_data['date'] = pd.to_datetime(fai_data['date'].astype(str), format='%Y-%m-%d')
    print(f"  原始数据: {len(fai_data)}条记录")

    # 清理行业名称
    fai_data['industry_clean'] = fai_data['industry'].str.replace('固定资产投资额累计同比增长率(%)', '')

    # 3. 读取映射配置
    print("\n3. 读取申万-NBS映射配置...")
    config_path = 'config/sw_nbs_mapping.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    sw_to_nbs = config['sw_to_nbs']
    print(f"  申万行业数量: {len(sw_to_nbs)}")

    # 4. 将NBS数据映射到申万行业
    print("\n4. 映射NBS数据到申万行业...")

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
    print(f"  申万行业PPI数据: {len(industry_ppi_df)}条记录")

    # 5. 将FAI数据映射到申万行业
    print("\n5. 映射FAI数据到申万行业...")

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
    print(f"  申万行业FAI数据: {len(industry_fai_df)}条记录")

    # 6. 计算PPI同比数据
    print("\n6. 计算PPI同比数据...")
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

    print(f"  PPI同比数据计算完成")

    # 7. 返回结果
    result = {
        'industry_ppi': industry_ppi_df,
        'industry_fai': industry_fai_df,
        'sw_industries': sw_industries
    }

    print("\n" + "=" * 80)
    print("数据加载完成")
    print("=" * 80)

    return result


if __name__ == '__main__':
    # 测试
    # 需要从2020年开始，因为计算同比需要12个月的数据
    result = load_nbs_industry_data(start_date='2020-01-01', end_date='2024-12-31')

    print("\n申万行业PPI数据（2024年Q4）:")
    q4_2024 = result['industry_ppi'][
        (result['industry_ppi']['date'].dt.quarter == 4) &
        (result['industry_ppi']['date'].dt.year == 2024)
    ]
    print(q4_2024[['sw_industry', 'date', 'ppi_yoy']].to_string(index=False))

    print("\n申万行业FAI数据（2024年Q4）:")
    q4_2024_fai = result['industry_fai'][
        (result['industry_fai']['date'].dt.quarter == 4) &
        (result['industry_fai']['date'].dt.year == 2024)
    ]
    print(q4_2024_fai[['sw_industry', 'date', 'fai_yoy']].to_string(index=False))