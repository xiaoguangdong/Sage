#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试adata库获取概念板块数据
"""

import adata
import pandas as pd

def test_adata_concept():
    """测试adata概念板块功能"""
    print("=" * 70)
    print("测试 adata 概念板块数据获取")
    print("=" * 70)
    
    # 1. 获取所有概念板块列表
    print("\n1. 获取概念板块列表...")
    try:
        concept_list = adata.stock.info.all_concept_code_ths()
        print(f"   ✓ 获取到 {len(concept_list)} 个概念板块")
        print(f"   \n前10个概念:")
        print(concept_list.head(10)[['index_code', 'concept']])
    except Exception as e:
        print(f"   ✗ 获取失败: {e}")
        return
    
    # 2. 获取热门概念的K线数据
    print("\n2. 获取热门概念的K线数据（锂电池、人工智能、新能源汽车）...")
    test_concepts = [
        ('885710', '锂电池'),
        ('885831', '人工智能'),
        ('885031', '新能源汽车')
    ]
    
    for index_code, concept_name in test_concepts:
        print(f"\n   测试 {concept_name} ({index_code})...")
        try:
            kline_data = adata.stock.market.get_market_concept_ths(
                index_code=index_code,
                k_type=1,  # 日线
                start_date='20240924',
                end_date='20241010'
            )
            print(f"   ✓ 数据量: {len(kline_data)} 条")
            if len(kline_data) > 0:
                print(f"   ✓ 最新数据 ({kline_data.iloc[-1]['trade_date']}):")
                print(f"     收盘: {kline_data.iloc[-1]['close']:.2f}")
                print(f"     涨跌幅: {kline_data.iloc[-1]['pct_change']:.2f}%")
                print(f"   ✓ 列名: {', '.join(kline_data.columns.tolist())}")
        except Exception as e:
            print(f"   ✗ 获取失败: {e}")
    
    # 3. 获取所有概念的实时行情
    print("\n3. 获取所有概念的实时行情...")
    try:
        real_time_all = adata.stock.market.get_market_concept_current_ths()
        print(f"   ✓ 获取到 {len(real_time_all)} 个概念的实时行情")
        print(f"   \n涨跌幅Top 10:")
        top10 = real_time_all.nlargest(10, 'pct_change')
        for idx, row in top10.iterrows():
            print(f"     {row['concept']}: {row['pct_change']:.2f}%")
    except Exception as e:
        print(f"   ✗ 获取失败: {e}")
    
    # 4. 获取资金流向数据
    print("\n4. 获取资金流向数据（近5日）...")
    try:
        capital_flow = adata.stock.market.all_capital_flow_east(day=5)
        print(f"   ✓ 获取到 {len(capital_flow)} 个概念的资金流向数据")
        print(f"   \n净流入Top 10:")
        top10_flow = capital_flow.nlargest(10, '净流入-5日')
        for idx, row in top10_flow.iterrows():
            print(f"     {row['概念名称']}: {row['净流入-5日']:.2f}亿")
    except Exception as e:
        print(f"   ✗ 获取失败: {e}")
    
    print("\n" + "=" * 70)
    print("✓ adata 测试完成")
    print("=" * 70)

if __name__ == '__main__':
    test_adata_concept()