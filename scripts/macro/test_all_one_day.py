#!/usr/bin/env python3
# -u
# -*- coding: utf-8 -*-

"""
测试Tushare sw_daily接口获取所有行业1个交易日的数据
"""

import pandas as pd
import tushare as ts
import time

from tushare_auth import get_tushare_token

def test_all_industries_one_day():
    """测试所有行业1个交易日的数据"""
    print("=" * 80)
    print("测试Tushare sw_daily接口")
    print("=" * 80)
    
    pro = ts.pro_api(get_tushare_token())
    
    # 获取所有申万行业指数列表
    print("\n获取申万行业指数列表...")
    indices = pro.index_classify(level='L1', src='SW2021')
    print(f"  获取到 {len(indices)} 个申万一级行业")
    
    # 测试获取所有行业某一天的数据
    print(f"\n测试获取所有行业2026-02-10的数据...")
    date = '20260210'
    
    all_data = []
    for i, row in indices.iterrows():
        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{len(indices)}")
        
        try:
            df = pro.sw_daily(ts_code=row['index_code'], start_date=date, end_date=date)
            if not df.empty:
                all_data.append(df)
            time.sleep(0.5)  # 避免频繁请求
        except Exception as e:
            print(f"  获取 {row['index_code']} 失败: {e}")
    
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        print(f"\n总记录数: {len(result_df)}")
        print(f"字段: {result_df.columns.tolist()}")
        print(f"\n数据预览:")
        print(result_df.to_string(index=False))
    else:
        print("\n未获取到数据")


if __name__ == '__main__':
    test_all_industries_one_day()
