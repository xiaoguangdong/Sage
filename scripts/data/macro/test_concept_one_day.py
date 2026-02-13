#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试概念板块接口单日数据量
"""

import tushare as ts
import time

from tushare_auth import get_tushare_token

token = get_tushare_token()
pro = ts.pro_api(token)

print("=" * 80)
print("测试概念板块接口单日数据量")
print("=" * 80)

# 使用2026-02-10（可能是最后一个交易日）
test_date = '20260210'

print(f"\n测试日期: {test_date}")

# 测试dc_index
print("\n=== dc_index (概念指数) ===")
try:
    offset = 0
    total_index = 0
    page = 1
    while True:
        df = pro.dc_index(trade_date=test_date, offset=offset)
        if df is None or df.empty:
            break
        total_index += len(df)
        print(f"  第{page}页: {len(df)} 条记录")
        offset += len(df)
        page += 1
        if len(df) < 5000:
            break
        time.sleep(40)
    print(f"  总计: {total_index} 条记录")
except Exception as e:
    print(f"  错误: {e}")

# 测试dc_member - 只获取第一页看看数据量
print("\n=== dc_member (概念成分股) ===")
print("只获取第一页估算数据量...")
try:
    df = pro.dc_member(trade_date=test_date, offset=0)
    print(f"  第1页: {len(df)} 条记录")
    if len(df) > 0:
        print(f"  唯一概念数: {df['ts_code'].nunique()}")
        print(f"  唯一成分股数: {df['con_code'].nunique()}")
        # 估算总页数
        if len(df) >= 5000:
            estimated_pages = len(df) // 5000 + 1
            estimated_total = len(df) * estimated_pages
            print(f"  估算总记录数: {estimated_total:,} 条")
        else:
            print(f"  总记录数: {len(df):,} 条")
except Exception as e:
    print(f"  错误: {e}")

print("\n" + "=" * 80)
