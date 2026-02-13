#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试Tushare概念板块接口用法
"""

import tushare as ts
import time

token = '2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e'
pro = ts.pro_api(token)

print("=" * 80)
print("测试Tushare概念板块接口用法")
print("=" * 80)

# 测试dc_index
print("\n=== dc_index (概念指数) ===")
print("获取概念指数列表，包括涨跌幅、换手率等")
try:
    # 先测试不带参数的
    df = pro.dc_index()
    print(f"不带参数记录数: {len(df)}")
    print(f"字段: {df.columns.tolist()}")
    print(f"唯一概念数: {df['ts_code'].nunique()}")
    print(f"日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"\n数据预览:")
    print(df.head(3).to_string(index=False))
    time.sleep(40)
except Exception as e:
    print(f"错误: {e}")

# 测试dc_member
print("\n=== dc_member (概念成分股) ===")
print("获取概念板块的成分股列表")
try:
    # 先测试不带参数的
    df = pro.dc_member()
    print(f"不带参数记录数: {len(df)}")
    print(f"字段: {df.columns.tolist()}")
    print(f"概念数量: {df['ts_code'].nunique()}")
    print(f"成分股数量: {df['con_code'].nunique()}")
    print(f"日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"\n数据预览:")
    print(df.head(3).to_string(index=False))
    time.sleep(40)
except Exception as e:
    print(f"错误: {e}")

# 测试dc_daily
print("\n=== dc_daily (概念成分股日线) ===")
print("获取概念成分股的日线行情")
try:
    # 先测试不带参数的
    df = pro.dc_daily()
    print(f"不带参数记录数: {len(df)}")
    print(f"字段: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"概念数量: {df['ts_code'].nunique()}")
        print(f"成分股数量: {df['con_code'].nunique()}")
        print(f"日期范围: {df['trade_date'].min()} {df['trade_date'].max()}")
        print(f"\n数据预览:")
        print(df.head(3).to_string(index=False))
except Exception as e:
    print(f"错误: {e}")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)