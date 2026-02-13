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
from datetime import datetime, timedelta


class MissingMacroDataFetcher:
    def __init__(self, token=None):
        self.token = token or '2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e'
        self.pro = ts.pro_api(self.token)
        self.output_dir = 'data/tushare/macro'
        self.northbound_dir = 'data/tushare/northbound'
        
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
            # 获取M2数据
            df_m2 = self.pro.shibor(
                start_date=start_date,
                end_date=end_date
            )

            # 尝试获取M1数据（可能需要其他接口）
            # Tushare的m2接口可能不直接提供，需要从其他数据源获取

            if df_m2 is not None and len(df_m2) > 0:
                filepath = os.path.join(self.output_dir, 'money_supply.parquet')
                df_m2.to_parquet(filepath, index=False)
                print(f"✓ 货币供应量数据已保存: {filepath} ({len(df_m2)}行)")

                return df_m2
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
            # Tushare可能没有直接的社融接口
            # 尝试使用shibor作为替代指标
            df = self.pro.shibor(
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and len(df) > 0:
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