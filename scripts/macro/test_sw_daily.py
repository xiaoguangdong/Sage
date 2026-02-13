#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Tushare sw_daily接口
获取申万行业日线行情数据
"""

import pandas as pd
import tushare as ts
import time
from datetime import datetime, timedelta

from tushare_auth import get_tushare_token

def test_sw_daily():
    """测试sw_daily接口"""
    print("=" * 80)
    print("测试Tushare sw_daily接口")
    print("=" * 80)
    
    pro = ts.pro_api(get_tushare_token())
    
    # 获取申万行业指数列表
    print("\n1. 获取申万行业指数列表...")
    try:
        index_df = pro.index_classify(level='L1', src='SW2021')
        print(f"  获取到 {len(index_df)} 个申万一级行业")
        print(f"  前5个行业:")
        print(index_df.head())
    except Exception as e:
        print(f"  获取失败: {e}")
        return
    
    # 测试sw_daily接口
    print("\n2. 测试sw_daily接口...")
    
    # 获取第一个行业的数据
    first_index = index_df.iloc[0]['index_code']
    print(f"  测试指数: {first_index}")
    
    try:
        # 获取最近一个月的数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        df = pro.sw_daily(ts_code=first_index, start_date=start_date, end_date=end_date)
        print(f"  获取到 {len(df)} 条记录")
        print(f"  字段列表: {df.columns.tolist()}")
        print(f"  前5行数据:")
        print(df.head())
        
        # 检查是否包含估值数据
        print(f"\n3. 检查是否包含估值数据...")
        if 'pe' in df.columns:
            print(f"  ✅ 包含PE数据")
        else:
            print(f"  ❌ 不包含PE数据")
        
        if 'pb' in df.columns:
            print(f"  ✅ 包含PB数据")
        else:
            print(f"  ❌ 不包含PB数据")
        
        if 'turnover_rate' in df.columns:
            print(f"  ✅ 包含换手率数据")
        else:
            print(f"  ❌ 不包含换手率数据")
        
    except Exception as e:
        print(f"  获取失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_sw_daily()
