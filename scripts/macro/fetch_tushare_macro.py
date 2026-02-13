#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tushare宏观数据获取脚本

功能：
1. 获取CPI数据
2. 获取PPI数据
3. 获取PMI数据
4. 获取10年期国债收益率
5. 支持断点续传（已获取的数据不会重复获取）
6. 支持重试机制（IP限制时自动重试）
"""

import tushare as ts
import pandas as pd
import os
import time
from datetime import datetime

from tushare_auth import get_tushare_token

class TushareMacroDataFetcher:
    def __init__(self, token):
        self.pro = ts.pro_api(token)
        self.output_dir = 'data/tushare/macro'
        os.makedirs(self.output_dir, exist_ok=True)
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 60  # 重试延迟60秒
    
    def load_existing_data(self, filename):
        """
        加载已存在的数据
        
        Args:
            filename: 文件名
        
        Returns:
            DataFrame: 已有数据，如果不存在返回None
        """
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 加载已有数据: {filename}")
            if filename.endswith('.parquet'):
                return pd.read_parquet(filepath)
            else:
                return pd.read_csv(filepath)
        return None
    
    def save_data(self, df, filename):
        """
        保存数据
        
        Args:
            df: DataFrame
            filename: 文件名
        """
        filepath = os.path.join(self.output_dir, filename)
        if filename.endswith('.parquet'):
            df.to_parquet(filepath, index=False)
        else:
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 数据已保存: {filename}")
    
    def fetch_cpi(self, start_m, end_m):
        """
        获取CPI数据（每批最多12个月）
        
        Args:
            start_m: 开始月份（格式：YYYYMM）
            end_m: 结束月份（格式：YYYYMM）
        
        Returns:
            DataFrame: CPI数据
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 获取CPI数据 ({start_m} ~ {end_m})...")
        
        for attempt in range(self.max_retries):
            try:
                df = self.pro.cn_cpi(start_m=start_m, end_m=end_m)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] CPI数据获取完成: {len(df)}条记录")
                time.sleep(60)  # 请求后等待60秒
                return df
            except Exception as e:
                if "IP数量超限" in str(e) and attempt < self.max_retries - 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] IP限制，第{attempt+1}次重试，等待{self.retry_delay}秒...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] CPI数据获取失败: {str(e)}")
                    return pd.DataFrame()
    
    def fetch_ppi(self, start_m, end_m):
        """
        获取PPI数据（每批最多12个月）
        
        Args:
            start_m: 开始月份（格式：YYYYMM）
            end_m: 结束月份（格式：YYYYMM）
        
        Returns:
            DataFrame: PPI数据
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 获取PPI数据 ({start_m} ~ {end_m})...")
        
        for attempt in range(self.max_retries):
            try:
                df = self.pro.cn_ppi(start_m=start_m, end_m=end_m)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] PPI数据获取完成: {len(df)}条记录")
                time.sleep(60)  # 请求后等待60秒
                return df
            except Exception as e:
                if "IP数量超限" in str(e) and attempt < self.max_retries - 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] IP限制，第{attempt+1}次重试，等待{self.retry_delay}秒...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] PPI数据获取失败: {str(e)}")
                    return pd.DataFrame()
    
    def fetch_pmi(self, start_m, end_m):
        """
        获取PMI数据（每批最多12个月）
        
        Args:
            start_m: 开始月份（格式：YYYYMM）
            end_m: 结束月份（格式：YYYYMM）
        
        Returns:
            DataFrame: PMI数据
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 获取PMI数据 ({start_m} ~ {end_m})...")
        
        for attempt in range(self.max_retries):
            try:
                df = self.pro.cn_pmi(start_m=start_m, end_m=end_m)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] PMI数据获取完成: {len(df)}条记录")
                time.sleep(60)  # 请求后等待60秒
                return df
            except Exception as e:
                if "IP数量超限" in str(e) and attempt < self.max_retries - 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] IP限制，第{attempt+1}次重试，等待{self.retry_delay}秒...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] PMI数据获取失败: {str(e)}")
                    return pd.DataFrame()
    
    def fetch_yield_curve(self, start_date, end_date):
        """
        获取国债收益率曲线
        
        Args:
            start_date: 开始日期（格式：YYYYMMDD）
            end_date: 结束日期（格式：YYYYMMDD）
        
        Returns:
            DataFrame: 收益率曲线数据
        """
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 获取国债收益率曲线...")
        
        for attempt in range(self.max_retries):
            try:
                df = self.pro.yc_cb(start_date=start_date, end_date=end_date)
                # 提取10年期收益率
                df_10y = df[['date', 'y10']].rename(columns={'y10': 'yield_10y'})
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 收益率曲线获取完成: {len(df_10y)}条记录")
                time.sleep(60)  # 请求后等待60秒
                return df_10y
            except Exception as e:
                if "IP数量超限" in str(e) and attempt < self.max_retries - 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] IP限制，第{attempt+1}次重试，等待{self.retry_delay}秒...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 收益率曲线获取失败: {str(e)}")
                    return pd.DataFrame()
    
    def fetch_all(self, start_date, end_date):
        """
        获取所有宏观数据（支持断点续传和重试）
        
        Args:
            start_date: 开始日期（格式：YYYY-MM-DD）
            end_date: 结束日期（格式：YYYY-MM-DD）
        
        Returns:
            dict: 包含所有数据的字典
        """
        print(f"=== 开始获取Tushare宏观数据 ===")
        print(f"时间范围: {start_date} ~ {end_date}")
        print(f"注意：每个接口请求后等待60秒以避免IP限制\n")
        
        # 转换日期格式
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 生成月份列表
        months = []
        current = start_dt
        while current <= end_dt:
            months.append(current.strftime('%Y%m'))
            current += pd.DateOffset(months=1)
        
        print(f"总时间范围: {len(months)}个月")
        
        # 分批获取（每批12个月）
        batch_size = 12
        batches = [months[i:i + batch_size] for i in range(0, len(months), batch_size)]
        
        print(f"分成 {len(batches)} 批，每批最多12个月")
        
        # 加载已有数据
        existing_cpi = self.load_existing_data('tushare_cpi.parquet')
        existing_ppi = self.load_existing_data('tushare_ppi.parquet')
        existing_pmi = self.load_existing_data('tushare_pmi.parquet')
        
        # 获取已有数据的月份
        existing_months_cpi = set(existing_cpi['month'].tolist()) if existing_cpi is not None and 'month' in existing_cpi.columns else set()
        existing_months_ppi = set(existing_ppi['month'].tolist()) if existing_ppi is not None and 'month' in existing_ppi.columns else set()
        existing_months_pmi = set(existing_pmi['month'].tolist()) if existing_pmi is not None and 'month' in existing_pmi.columns else set()
        
        print(f"已有数据: CPI {len(existing_months_cpi)}个月, PPI {len(existing_months_ppi)}个月, PMI {len(existing_months_pmi)}个月")
        
        all_cpi = [existing_cpi] if existing_cpi is not None else []
        all_ppi = [existing_ppi] if existing_ppi is not None else []
        all_pmi = [existing_pmi] if existing_pmi is not None else []
        
        for i, batch in enumerate(batches, 1):
            print(f"\n=== 批次 {i}/{len(batches)} ===")
            print(f"获取月份: {batch[0]} 到 {batch[-1]}")
            
            # 检查是否已有数据
            batch_set = set(batch)
            
            # CPI
            if batch_set.issubset(existing_months_cpi):
                print(f"  CPI: 已有数据，跳过")
                cpi_batch = None
            else:
                missing_months = sorted(batch_set - existing_months_cpi)
                if missing_months:
                    print(f"  CPI: 需要获取 {len(missing_months)}个月: {missing_months[0]} ~ {missing_months[-1]}")
                    if len(missing_months) == 1:
                        cpi_batch = self.fetch_cpi(missing_months[0], missing_months[0])
                    else:
                        cpi_batch = self.fetch_cpi(missing_months[0], missing_months[-1])
                else:
                    cpi_batch = None
            
            # PPI
            if batch_set.issubset(existing_months_ppi):
                print(f"  PPI: 已有数据，跳过")
                ppi_batch = None
            else:
                missing_months = sorted(batch_set - existing_months_ppi)
                if missing_months:
                    print(f"  PPI: 需要获取 {len(missing_months)}个月: {missing_months[0]} ~ {missing_months[-1]}")
                    if len(missing_months) == 1:
                        ppi_batch = self.fetch_ppi(missing_months[0], missing_months[0])
                    else:
                        ppi_batch = self.fetch_ppi(missing_months[0], missing_months[-1])
                else:
                    ppi_batch = None
            
            # PMI
            if batch_set.issubset(existing_months_pmi):
                print(f"  PMI: 已有数据，跳过")
                pmi_batch = None
            else:
                missing_months = sorted(batch_set - existing_months_pmi)
                if missing_months:
                    print(f"  PMI: 需要获取 {len(missing_months)}个月: {missing_months[0]} ~ {missing_months[-1]}")
                    if len(missing_months) == 1:
                        pmi_batch = self.fetch_pmi(missing_months[0], missing_months[0])
                    else:
                        pmi_batch = self.fetch_pmi(missing_months[0], missing_months[-1])
                else:
                    pmi_batch = None
            
            # 添加新数据
            if cpi_batch is not None and len(cpi_batch) > 0:
                all_cpi.append(cpi_batch)
            if ppi_batch is not None and len(ppi_batch) > 0:
                all_ppi.append(ppi_batch)
            if pmi_batch is not None and len(pmi_batch) > 0:
                all_pmi.append(pmi_batch)
        
        # 合并并去重数据
        cpi = pd.concat(all_cpi, ignore_index=True).drop_duplicates(subset=['month']) if all_cpi else pd.DataFrame()
        ppi = pd.concat(all_ppi, ignore_index=True).drop_duplicates(subset=['month']) if all_ppi else pd.DataFrame()
        pmi = pd.concat(all_pmi, ignore_index=True).drop_duplicates(subset=['month']) if all_pmi else pd.DataFrame()
        
        # 获取收益率曲线
        print(f"\n=== 获取收益率曲线 ===")
        start_date_str = start_dt.strftime('%Y%m%d')
        end_date_str = end_dt.strftime('%Y%m%d')
        yield_curve = self.fetch_yield_curve(start_date_str, end_date_str)
        
        # 保存数据
        print(f"\n=== 保存数据 ===")
        if len(cpi) > 0:
            self.save_data(cpi, 'tushare_cpi.parquet')
        if len(ppi) > 0:
            self.save_data(ppi, 'tushare_ppi.parquet')
        if len(pmi) > 0:
            self.save_data(pmi, 'tushare_pmi.parquet')
        if len(yield_curve) > 0:
            self.save_data(yield_curve, 'tushare_yield_10y.parquet')
        
        print(f"\n=== 获取完成 ===")
        print(f"CPI: {len(cpi)}条记录")
        print(f"PPI: {len(ppi)}条记录")
        print(f"PMI: {len(pmi)}条记录")
        print(f"收益率曲线: {len(yield_curve)}条记录")
        
        return {
            'cpi': cpi,
            'ppi': ppi,
            'pmi': pmi,
            'yield_curve': yield_curve
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='获取Tushare宏观数据')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='开始日期（格式：YYYY-MM-DD）')
    parser.add_argument('--end-date', type=str, default=None, help='结束日期（格式：YYYY-MM-DD），默认为今天')
    
    args = parser.parse_args()
    
    # 如果没有指定结束日期，使用今天
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    token = get_tushare_token()
    
    # 创建fetcher
    fetcher = TushareMacroDataFetcher(token)
    
    # 获取数据
    data = fetcher.fetch_all(args.start_date, args.end_date)


if __name__ == '__main__':
    main()
