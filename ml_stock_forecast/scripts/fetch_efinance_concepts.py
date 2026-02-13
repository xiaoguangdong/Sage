#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 efinance 获取概念板块数据
"""

import efinance as ef
import pandas as pd
import time
from typing import Optional, List
import os

def get_all_concept_list() -> Optional[pd.DataFrame]:
    """获取所有概念板块列表"""
    print("获取所有概念板块列表...")
    
    # 通过获取热门股票所属板块来提取概念板块
    # 由于 efinance 没有直接获取所有概念板块的接口
    # 我们可以通过获取代表性股票的板块来收集概念板块
    
    # 一些热门股票代码
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
    ]
    
    all_boards = set()
    for stock_code in sample_stocks:
        try:
            df = ef.stock.get_belong_board(stock_code)
            if df is not None and len(df) > 0:
                for _, row in df.iterrows():
                    board_name = row['板块名称']
                    board_code = row['板块代码']
                    # 判断是否是概念板块（概念板块通常不是行业）
                    if not any(keyword in board_name for keyword in ['板块', '地区', '地域']):
                        all_boards.add((board_code, board_name))
            time.sleep(0.5)  # 避免请求过快
        except Exception as e:
            print(f"  获取 {stock_code} 失败: {e}")
    
    if all_boards:
        result = pd.DataFrame(list(all_boards), columns=['板块代码', '板块名称'])
        print(f"✓ 收集到 {len(result)} 个概念板块")
        return result
    else:
        print("✗ 未收集到任何概念板块")
        return None

def get_concept_history(board_code: str, board_name: str, 
                       start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """获取概念板块历史K线数据"""
    try:
        df = ef.stock.get_quote_history(board_code, beg=start_date, end=end_date)
        if df is not None and len(df) > 0:
            df['板块名称'] = board_name
            return df
    except Exception as e:
        print(f"  ✗ {board_name} 获取失败: {e}")
    return None

def fetch_all_concepts_history(start_date='20200101', end_date=None, max_concepts=50):
    """获取所有概念板块的历史数据"""
    if end_date is None:
        from datetime import datetime
        end_date = datetime.now().strftime('%Y%m%d')
    
    print(f"开始下载概念板块历史数据 ({start_date} ~ {end_date})...")
    
    # 获取概念板块列表
    concept_list = get_all_concept_list()
    if concept_list is None:
        return None
    
    # 限制数量避免请求过多
    concept_list = concept_list.head(max_concepts)
    
    all_data = []
    for idx, row in concept_list.iterrows():
        board_code = row['板块代码']
        board_name = row['板块名称']
        
        print(f"[{idx+1}/{len(concept_list)}] 下载 {board_name} ({board_code})...")
        
        df = get_concept_history(board_code, board_name, start_date, end_date)
        if df is not None:
            all_data.append(df)
        
        # 避免请求过快
        time.sleep(0.5)
    
    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        print(f"\n✓ 总计下载 {len(result)} 条数据")
        return result
    else:
        return None

if __name__ == '__main__':
    # 测试获取概念列表
    print("测试获取概念板块列表...")
    concepts = get_all_concept_list()
    if concepts is not None:
        print(concepts.head(20))
        
        # 测试获取历史数据
        print("\n测试获取锂电池概念历史数据...")
        lithium = concepts[concepts['板块名称'] == '锂电池'].iloc[0]
        history = get_concept_history(
            lithium['板块代码'],
            lithium['板块名称'],
            '20240901',
            '20241010'
        )
        if history is not None:
            print(f"✓ 成功获取 {len(history)} 条数据")
            print(history.head())
    else:
        print("✗ 测试失败")