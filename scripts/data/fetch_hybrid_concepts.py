#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合数据源方案：Tushare获取概念列表 + efinance获取概念K线
结合两者的优势，实现最佳性能和功能
"""

import tushare as ts
import efinance as ef
import pandas as pd
import time
from typing import Optional, Dict, List
import os

# Tushare token
TUSHARE_TOKEN = '2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e'

def get_tushare_concept_list() -> pd.DataFrame:
    """从Tushare获取完整的概念列表"""
    print("从 Tushare 获取概念列表...")
    pro = ts.pro_api(TUSHARE_TOKEN)
    
    df = pro.concept()
    print(f"✓ 成功获取 {len(df)} 个概念板块")
    return df

def get_efinance_concept_kline(concept_name: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """从efinance获取概念K线数据"""
    try:
        # efinance需要板块代码，先查找
        # 通过代表性股票查找概念代码
        sample_stocks = ['000001', '000002', '600000', '600519']
        
        for stock_code in sample_stocks:
            try:
                boards = ef.stock.get_belong_board(stock_code)
                if boards is not None and len(boards) > 0:
                    target = boards[boards['板块名称'] == concept_name]
                    if len(target) > 0:
                        board_code = target.iloc[0]['板块代码']
                        # 获取K线
                        df = ef.stock.get_quote_history(board_code, beg=start_date, end=end_date)
                        if df is not None and len(df) > 0:
                            df['板块名称'] = concept_name
                            return df
            except:
                continue
        
        return None
    except Exception as e:
        print(f"  ✗ {concept_name} 获取失败: {e}")
        return None

def build_concept_code_mapping(tushare_concepts: pd.DataFrame) -> Dict[str, str]:
    """
    建立Tushare概念名称到efinance板块代码的映射
    
    通过获取代表性股票的板块信息，建立映射关系
    """
    print("建立概念名称到板块代码的映射...")
    
    # 一些热门股票代码，覆盖多个概念
    sample_stocks = [
        '002466',  # 天齐锂业
        '300014',  # 亿纬锂能
        '000858',  # 五粮液
        '600519',  # 贵州茅台
        '300750',  # 宁德时代
        '601888',  # 中国中免
        '002594',  # 比亚迪
        '600276',  # 恒瑞医药
        '300760',  # 迈瑞医疗
        '688981',  # 中芯国际
        '000001',  # 平安银行
        '000002',  # 万科A
        '600000',  # 浦发银行
        '600036',  # 招商银行
        '600519',  # 贵州茅台
        '000858',  # 五粮液
        '002415',  # 海康威视
        '300059',  # 东方财富
        '600030',  # 中信证券
        '601318',  # 中国平安
    ]
    
    mapping = {}
    
    for stock_code in sample_stocks:
        try:
            boards = ef.stock.get_belong_board(stock_code)
            if boards is not None and len(boards) > 0:
                for _, row in boards.iterrows():
                    board_name = row['板块名称']
                    board_code = row['板块代码']
                    # 检查是否在Tushare概念列表中
                    if board_name in tushare_concepts['name'].values:
                        mapping[board_name] = board_code
            time.sleep(0.3)  # 避免请求过快
        except Exception as e:
            continue
    
    print(f"✓ 建立了 {len(mapping)} 个概念映射")
    return mapping

def fetch_all_concepts_history(start_date='20200101', end_date=None, max_concepts=100):
    """
    获取所有概念板块的历史数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        max_concepts: 最多获取的概念数量
    """
    if end_date is None:
        from datetime import datetime
        end_date = datetime.now().strftime('%Y%m%d')
    
    print(f"混合数据源方案：Tushare概念列表 + efinance概念K线")
    print(f"时间范围: {start_date} ~ {end_date}")
    print("=" * 70)
    
    # 步骤1: 从Tushare获取完整概念列表
    tushare_concepts = get_tushare_concept_list()
    if tushare_concepts is None or len(tushare_concepts) == 0:
        print("✗ 未能获取概念列表")
        return None
    
    # 限制数量
    tushare_concepts = tushare_concepts.head(max_concepts)
    print(f"\n将获取前 {len(tushare_concepts)} 个概念的历史数据")
    
    # 步骤2: 建立概念名称到efinance板块代码的映射
    mapping = build_concept_code_mapping(tushare_concepts)
    
    # 步骤3: 获取概念K线数据
    print(f"\n开始下载概念K线数据...")
    all_data = []
    
    for idx, row in tushare_concepts.iterrows():
        concept_name = row['name']
        concept_code = row['code']
        
        print(f"[{idx+1}/{len(tushare_concepts)}] {concept_name} ({concept_code})...", end=' ')
        
        if concept_name in mapping:
            board_code = mapping[concept_name]
            try:
                df = ef.stock.get_quote_history(board_code, beg=start_date, end=end_date)
                if df is not None and len(df) > 0:
                    df['概念名称'] = concept_name
                    df['概念代码'] = concept_code
                    all_data.append(df)
                    print(f"✓ {len(df)} 条")
                else:
                    print("✗ 无数据")
            except Exception as e:
                print(f"✗ 失败: {e}")
        else:
            print("⊗ 未找到板块代码")
        
        # 避免请求过快
        time.sleep(0.3)
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ 总计下载 {len(result)} 条数据")
        print(f"✓ 覆盖 {len(all_data)} 个概念板块")
        return result
    else:
        print("\n✗ 未获取到任何数据")
        return None

if __name__ == '__main__':
    # 测试混合方案
    print("测试混合数据源方案...")
    
    # 测试获取概念列表
    concepts = get_tushare_concept_list()
    print(f"\n前10个概念:")
    print(concepts.head(10))
    
    # 测试建立映射
    print("\n测试建立映射...")
    mapping = build_concept_code_mapping(concepts)
    print(f"\n前10个映射:")
    for i, (name, code) in enumerate(list(mapping.items())[:10]):
        print(f"  {name} -> {code}")
    
    # 测试获取K线
    print("\n测试获取概念K线...")
    if len(mapping) > 0:
        test_concept = list(mapping.keys())[0]
        test_code = mapping[test_concept]
        print(f"测试: {test_concept} ({test_code})")
        
        df = ef.stock.get_quote_history(test_code, beg='20240901', end='20241010')
        if df is not None and len(df) > 0:
            print(f"✓ 成功获取 {len(df)} 条数据")
            print(df.head())
        else:
            print("✗ 无数据")