#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试Tushare sw_daily接口获取单个行业1个月的数据
"""

import pandas as pd
import tushare as ts
import time

from tushare_auth import get_tushare_token

def test_one_industry_one_month():
    """测试1个行业1个月的数据"""
    print("=" * 80)
    print("测试Tushare sw_daily接口")
    print("=" * 80)
    
    pro = ts.pro_api(get_tushare_token())
    
    # 测试农林牧渔（801010.SI）
    print("\n测试1个行业1个月的数据:")
    ts_code = '801010.SI'
    start_date = '20260101'
    end_date = '20260131'
    
    try:
        df = pro.sw_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        print(f"  指数: {ts_code}")
        print(f"  日期范围: {start_date} ~ {end_date}")
        print(f"  获取到 {len(df)} 条记录")
        print(f"  字段: {df.columns.tolist()}")
        print()
        print(f"  数据预览:")
        print(df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"  获取失败: {e}")


if __name__ == '__main__':
    test_one_industry_one_month()
