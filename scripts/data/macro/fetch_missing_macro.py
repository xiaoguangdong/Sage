#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
获取缺失的宏观数据

根据《宏观经济预测模型设计文档》要求，还需要获取：
1. 10年期国债收益率
2. 社融存量（M2）
3. M1/M2货币供应量
4. 北向资金流向
5. 北向资金持仓
"""

import tushare as ts
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.tushare_helpers import get_pro
from scripts.data.macro.paths import MACRO_DIR, NORTHBOUND_DIR

class MissingMacroDataFetcher:
    def __init__(self, token=None):
        self.pro = get_pro(token)
        self.output_dir = str(MACRO_DIR)
        self.northbound_dir = str(NORTHBOUND_DIR)
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.northbound_dir, exist_ok=True)
    
    def fetch_yield_curve(self, start_date='20200101', end_date='20251231'):
        """
        获取国债收益率曲线

        主要关注10年期国债收益率
        """
        print("\n=== 获取国债收益率曲线 ===")
        
        try:
            df = self.pro.bz_yield(
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and len(df) > 0:
                # 提取10年期收益率
                df_y10 = df[['date', 'y10']].rename(columns={'y10': 'yield10y'})
                
                # 保存
                filepath = os.path.join(self.output_dir, 'yield_curve.parquet')
                df_y10.to_parquet(filepath, index=False)
                print(f"✓ 10年期国债收益率已保存: {filepath} ({len(df_y10)}行)")

                return df_y10
        except Exception as e:
            print(f"获取国债收益率失败: {e}")

        return None
    
    def fetch_money_supply(self, start_date='20200101', end_date='20251231'):
        """
        获取货币供应量数据（M1/M2）

        用于计算M1-M2剪刀差
        """
        print("\n=== 获取货币供应量数据 ===")
        
        try:
            start_m = pd.to_datetime(start_date).strftime('%Y%m')
            end_m = pd.to_datetime(end_date).strftime('%Y%m')

            df_m = self.pro.cn_m(start_m=start_m, end_m=end_m)
            if df_m is not None and len(df_m) > 0:
                df_m['date'] = pd.to_datetime(df_m['month'].astype(str), format='%Y%m')
                df_m = df_m.rename(columns={
                    'm1_yoy': 'm1_yoy',
                    'm2_yoy': 'm2_yoy'
                })
                filepath = os.path.join(self.output_dir, 'money_supply.parquet')
                df_m.to_parquet(filepath, index=False)
                print(f"✓ 货币供应量数据已保存: {filepath} ({len(df_m)}行)")

                return df_m
        except Exception as e:
            print(f"获取货币供应量数据失败: {e}")

        return None
    
    def fetch_credit_data(self, start_date='20200101', end_date='20251231'):
        """
        获取信用数据（社融存量）

        用于判断信用环境
        """
        print("\n=== 获取信用数据（社融存量）===")
        
        try:
            start_m = pd.to_datetime(start_date).strftime('%Y%m')
            end_m = pd.to_datetime(end_date).strftime('%Y%m')

            df = self.pro.sf_month(start_m=start_m, end_m=end_m)
            if df is not None and len(df) > 0:
                df['month'] = df['month'].astype(str)
                df = df.sort_values('month')
                df['date'] = pd.to_datetime(df['month'], format='%Y%m')
                df['stk_endval'] = pd.to_numeric(df['stk_endval'], errors='coerce')
                df['credit_growth'] = df['stk_endval'].pct_change(12) * 100

                filepath = os.path.join(self.output_dir, 'credit_data.parquet')
                df.to_parquet(filepath, index=False)
                print(f"✓ 信用数据已保存: {filepath} ({len(df)}行)")

                return df
        except Exception as e:
            print(f"获取信用数据失败: {e}")

        return None
    
    def fetch_northbound_flow(self, start_date='20200101', end_date='20251231'):
        """
        获取北向资金流向数据

        接口：moneyflow_hsgt
        """
        print("\n=== 获取北向资金流向数据 ===")
        
        try:
            df = self.pro.moneyflow_hsgt(
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and len(df) > 0:
                # 保存
                filepath = os.path.join(self.northbound_dir, 'daily_flow.parquet')
                df.to_parquet(filepath, index=False)
                print(f"✓ 北向资金流向数据已保存: {filepath} ({len(df)}行)")

                return df
        except Exception as e:
            print(f"获取北向资金流向数据失败: {e}")

        return None
    
    def fetch_northbound_holdings(self, start_date='20200101', end_date='20251231'):
        """
        获取北向资金持仓数据

        接口：hk_hold
        """
        print("\n=== 获取北向资金持仓数据 ===")
        
        try:
            df = self.pro.hk_hold(
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and len(df) > 0:
                # 保存
                filepath = os.path.join(self.northbound_dir, 'hk_hold.parquet')
                df.to_parquet(filepath, index=False)
                print(f"✓ 北向资金持仓数据已保存: {filepath} ({len(df)}行)")

                return df
        except Exception as e:
            print(f"获取北向资金持仓数据失败: {e}")

        return None
    
    def fetch_all(self, start_date='20200101', end_date='20251231'):
        """
        获取所有缺失的宏观数据
        """
        print("=" * 80)
        print("开始获取缺失的宏观数据")
        print(f"时间范围: {start_date} 至 {end_date}")
        print("=" * 80)

        # 1. 10年期国债收益率
        self.fetch_yield_curve(start_date, end_date)

        # 2. 货币供应量（M1/M2）
        self.fetch_money_supply(start_date, end_date)

        # 3. 信用数据
        self.fetch_credit_data(start_date, end_date)

        # 4. 北向资金流向
        self.fetch_northbound_flow(start_date, end_date)

        # 5. 北向资金持仓
        self.fetch_northbound_holdings(start_date, end_date)

        print("\n" + "=" * 80)
        print("所有缺失数据获取完成！")
        print("=" * 80)


def main():
    fetcher = MissingMacroDataFetcher()
    fetcher.fetch_all(start_date='20200101', end_date='20251231')


if __name__ == '__main__':
    main()
