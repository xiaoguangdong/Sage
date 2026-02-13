#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
北向资金数据获取脚本

功能：
1. 获取北向资金日度流向
2. 获取北向资金持仓数据
3. 计算行业持仓分布

数据来源：Tushare Pro
"""

import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
import os
import time

from tushare_auth import get_tushare_token
class NorthboundDataFetcher:
    """北向资金数据获取器"""
    
    def __init__(self, token):
        self.pro = ts.pro_api(token)
        self.output_dir = 'data/tushare/northbound'
        os.makedirs(self.output_dir, exist_ok=True)
        self.api_delay = 30  # API请求间隔30秒
    
    def fetch_daily_flow(self, start_date, end_date):
        """
        获取北向资金日度流向
        
        Args:
            start_date: 开始日期（格式：YYYYMMDD）
            end_date: 结束日期（格式：YYYYMMDD）
        
        Returns:
            DataFrame: 日度流向数据
        """
        print("获取北向资金日度流向...")
        df = self.pro.moneyflow_hsgt(start_date=start_date, end_date=end_date)
        time.sleep(self.api_delay)  # 请求后等待30秒
        return df
    
    def fetch_hk_hold(self, start_date, end_date):
        """
        获取北向资金持仓数据
        
        Args:
            start_date: 开始日期（格式：YYYYMMDD）
            end_date: 结束日期（格式：YYYYMMDD）
        
        Returns:
            DataFrame: 持仓数据
        """
        print("获取北向资金持仓数据...")
        df = self.pro.hk_hold(start_date=start_date, end_date=end_date)
        time.sleep(self.api_delay)  # 请求后等待30秒
        return df
    
    def fetch_all(self, start_date, end_date):
        """
        获取所有北向资金数据
        
        Args:
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）
        
        Returns:
            dict: 包含所有数据的字典
        """
        print(f"=== 开始获取北向资金数据 ===")
        print(f"时间范围: {start_date} ~ {end_date}\n")
        
        # 转换日期格式
        start_date_int = start_date.replace('-', '')
        end_date_int = end_date.replace('-', '')
        
        # 获取数据
        daily_flow = self.fetch_daily_flow(start_date_int, end_date_int)
        hk_hold = self.fetch_hk_hold(start_date_int, end_date_int)
        
        # 保存数据
        daily_flow_path = f"{self.output_dir}/northbound_daily_flow.parquet"
        hk_hold_path = f"{self.output_dir}/northbound_hk_hold.parquet"
        
        daily_flow.to_parquet(daily_flow_path, index=False)
        hk_hold.to_parquet(hk_hold_path, index=False)
        
        print(f"\n✓ 日度流向数据已保存: {daily_flow_path}")
        print(f"✓ 持仓数据已保存: {hk_hold_path}")
        
        return {
            'daily_flow': daily_flow,
            'hk_hold': hk_hold
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='北向资金数据获取脚本')
    parser.add_argument('--start-date', type=str, default=None, help='开始日期（YYYY-MM-DD），默认为1个月前')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期（YYYY-MM-DD），默认为今天')
    
    args = parser.parse_args()
    
    # 默认获取最近1个月的数据
    if args.start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
    
    if args.end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    fetcher = NorthboundDataFetcher(get_tushare_token())
    data = fetcher.fetch_all(start_date, end_date)
    
    print(f"\n=== 数据获取完成 ===")


if __name__ == '__main__':
    main()
