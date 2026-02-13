#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试Tushare sw_daily接口获取单个行业1个交易日的数据
"""

import pandas as pd
import tushare as ts
import time


def test_one_day_one_industry():
    """测试1个行业1个交易日的数据"""
    print("=" * 80)
    print("测试Tushare sw_daily接口")
    print("=" * 80)
    
    # 设置Tushare token
    token = '2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e'
    pro = ts.pro_api(token)
    
    # 测试农林牧渔（801010.SI）
    print("\n测试1个行业1个交易日:")
    ts_code = '801010.SI'
    date = '20260210'
    
    try:
        df = pro.sw_daily(ts_code=ts_code, start_date=date, end_date=date)
        print(f"  指数: {ts_code}")
        print(f"  日期: {date}")
        print(f"  获取到 {len(df)} 条记录")
        print(f"  字段: {df.columns.tolist()}")
        print()
        print(f"  数据:")
        print(df.to_string(index=False))
        
    except Exception as e:
        print(f"  获取失败: {e}")


if __name__ == '__main__':
    test_one_day_one_industry()