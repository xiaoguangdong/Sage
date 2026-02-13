#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用Baostock下载A股历史K线数据 (2020-01-01 到 2024-02-05)
"""

import baostock as bs
import pandas as pd
import os
from datetime import datetime
import time

# 配置参数
START_DATE = '2020-01-01'
END_DATE = '2024-02-05'
DATA_DIR = './data/baostock'
FREQUENCY = 'd'  # 日线数据

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)


def get_all_stocks():
    """获取所有A股股票代码"""
    rs = bs.query_all_stock(day=END_DATE)
    stock_list = []
    while (rs.error_code == '0') & rs.next():
        stock_list.append(rs.get_row_data())
    
    df = pd.DataFrame(stock_list, columns=rs.fields)
    
    # 过滤只保留A股
    df = df[df['type'] == '1']  # type='1' 表示股票
    return df[['code', 'code_name', 'ipoDate']]


def download_stock_data(stock_code, start_date, end_date):
    """下载单只股票的历史数据"""
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"  # "3": 不复权
    )
    
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    if not data_list:
        return None
    
    df = pd.DataFrame(data_list, columns=rs.fields)
    return df


def main():
    print("="*70)
    print("Baostock A股数据下载工具")
    print("="*70)
    print(f"日期范围: {START_DATE} 至 {END_DATE}")
    print(f"数据目录: {DATA_DIR}")
    print()
    
    # 登录系统
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return
    
    print("✓ Baostock登录成功")
    print()
    
    # 获取所有股票列表
    print("正在获取股票列表...")
    stocks_df = get_all_stocks()
    print(f"✓ 共找到 {len(stocks_df)} 只A股")
    print()
    
    # 统计信息
    success_count = 0
    fail_count = 0
    total_stocks = len(stocks_df)
    
    # 开始下载
    print("开始下载K线数据...")
    print("-"*70)
    
    for idx, row in stocks_df.iterrows():
        stock_code = row['code']
        stock_name = row['code_name']
        
        print(f"[{idx+1}/{total_stocks}] 下载 {stock_code} {stock_name}...", end=' ', flush=True)
        
        # 下载数据
        df = download_stock_data(stock_code, START_DATE, END_DATE)
        
        if df is not None and len(df) > 0:
            # 保存到parquet文件
            file_path = os.path.join(DATA_DIR, f"{stock_code}.parquet")
            df.to_parquet(file_path, index=False)
            
            success_count += 1
            print(f"✓ ({len(df)}条)")
        else:
            fail_count += 1
            print("✗ 无数据")
        
        # 避免请求过快
        time.sleep(0.1)
    
    # 登出系统
    bs.logout()
    
    print()
    print("="*70)
    print("下载完成!")
    print(f"成功: {success_count} 只股票")
    print(f"失败: {fail_count} 只股票")
    print(f"数据保存位置: {DATA_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()