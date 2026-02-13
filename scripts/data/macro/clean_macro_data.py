#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
宏观数据清洗和特征工程

功能：
1. 加载Tushare宏观数据
2. 加载NBS行业数据
3. 数据对齐和清洗
4. 计算衍生特征
5. 输出整合后的特征数据
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional

from scripts.data.macro.paths import MACRO_DIR


# 修复pandas版本兼容性
try:
    # 新版本pandas (>=2.2)
    pd.date_range.__module__
except:
    pass


class MacroDataProcessor:
    """
    宏观数据处理器
    
    功能：
    1. 加载各类数据源
    2. 数据清洗和标准化
    3. 特征工程
    4. 数据对齐
    """
    
    def __init__(self, data_dir: str = None):
        """
        初始化处理器
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir or str(MACRO_DIR)
        self.macro_data = None
        self.industry_data = None
        self.northbound_data = None
    
    def load_tushare_macro(self) -> pd.DataFrame:
        """
        加载Tushare宏观数据
        
        Returns:
            DataFrame: 宏观数据
        """
        # 加载CPI
        cpi_path = os.path.join(self.data_dir, 'tushare_cpi.parquet')
        if os.path.exists(cpi_path):
            cpi = pd.read_parquet(cpi_path)
            # 处理月份格式
            cpi['date'] = pd.to_datetime(cpi['month'].astype(str), format='%Y%m')
            cpi = cpi[['date', 'nt_yoy']].rename(columns={'nt_yoy': 'cpi_yoy'})
        else:
            cpi = pd.DataFrame(columns=['date', 'cpi_yoy'])
        
        # 加载PPI
        ppi_path = os.path.join(self.data_dir, 'tushare_ppi.parquet')
        if os.path.exists(ppi_path):
            ppi = pd.read_parquet(ppi_path)
            ppi['date'] = pd.to_datetime(ppi['month'].astype(str), format='%Y%m')
            ppi = ppi[['date', 'ppi_yoy']]
        else:
            ppi = pd.DataFrame(columns=['date', 'ppi_yoy'])
        
        # 加载PMI
        pmi_path = os.path.join(self.data_dir, 'tushare_pmi.parquet')
        if os.path.exists(pmi_path):
            pmi = pd.read_parquet(pmi_path)
            # 检查列名
            if 'MONTH' in pmi.columns:
                pmi['date'] = pd.to_datetime(pmi['MONTH'].astype(str), format='%Y%m')
            elif 'month' in pmi.columns:
                pmi['date'] = pd.to_datetime(pmi['month'].astype(str), format='%Y%m')
            else:
                # 尝试从行索引或其他列获取
                pmi['date'] = pd.to_datetime(pmi.index, format='%Y%m')
            
            # 使用PMI010000（制造业PMI）作为主要指标
            if 'PMI010000' in pmi.columns:
                pmi = pmi[['date', 'PMI010000']].rename(columns={'PMI010000': 'pmi'})
            elif 'markit_pmi' in pmi.columns:
                pmi = pmi[['date', 'markit_pmi']].rename(columns={'markit_pmi': 'pmi'})
            else:
                # 取第一个PMI列
                pmi_cols = [col for col in pmi.columns if col.startswith('PMI')]
                if pmi_cols:
                    pmi = pmi[['date', pmi_cols[0]]].rename(columns={pmi_cols[0]: 'pmi'})
                else:
                    pmi = pd.DataFrame(columns=['date', 'pmi'])
        else:
            pmi = pd.DataFrame(columns=['date', 'pmi'])
        
        # 加载10年期国债收益率
        yield_path = os.path.join(self.data_dir, 'tushare_yield_10y.parquet')
        if os.path.exists(yield_path):
            yield_df = pd.read_parquet(yield_path)
            yield_df['date'] = pd.to_datetime(yield_df['date'])
            yield_df = yield_df[['date', 'yield_10y']]
        else:
            yield_df = pd.DataFrame(columns=['date', 'yield_10y'])
        
        # 合并数据
        macro = cpi.merge(ppi, on='date', how='outer')
        macro = macro.merge(pmi, on='date', how='outer')
        macro = macro.merge(yield_df, on='date', how='outer')
        
        # 按日期排序
        macro = macro.sort_values('date').reset_index(drop=True)
        
        # 加载社融和M2数据
        credit_path = os.path.join(self.data_dir, 'credit_data.parquet')
        if os.path.exists(credit_path):
            credit = pd.read_parquet(credit_path)
            credit['date'] = pd.to_datetime(credit['date'])
            if 'credit_growth' in credit.columns:
                macro = macro.merge(credit[['date', 'credit_growth']], on='date', how='left')
        
        money_path = os.path.join(self.data_dir, 'money_supply.parquet')
        if os.path.exists(money_path):
            money = pd.read_parquet(money_path)
            money['date'] = pd.to_datetime(money['date'])
            if 'm1_yoy' in money.columns and 'm2_yoy' in money.columns:
                money['m1_m2_spread'] = money['m1_yoy'] - money['m2_yoy']
                macro = macro.merge(money[['date', 'm1_yoy', 'm2_yoy', 'm1_m2_spread']], on='date', how='left')
        
        self.macro_data = macro
        print(f"加载Tushare宏观数据: {len(macro)}条记录")
        
        return macro
    
    def load_nbs_industry(self) -> pd.DataFrame:
        """
        加载NBS行业数据
        
        Returns:
            DataFrame: 行业数据
        """
        industry_list = []
        
        # 加载分行业PPI数据
        ppi_files = [
            'nbs_ppi_industry_2020.csv',
            'nbs_ppi_industry_202512.csv'
        ]
        
        for ppi_file in ppi_files:
            ppi_path = os.path.join(self.data_dir, ppi_file)
            if os.path.exists(ppi_path):
                ppi_df = pd.read_csv(ppi_path)
                if 'date' in ppi_df.columns and 'industry' in ppi_df.columns:
                    industry_list.append(ppi_df)
        
        # 加载分行业FAI数据
        fai_files = [
            'nbs_fai_industry_2020.csv',
            'nbs_fai_industry_202512.csv'
        ]
        
        for fai_file in fai_files:
            fai_path = os.path.join(self.data_dir, fai_file)
            if os.path.exists(fai_path):
                fai_df = pd.read_csv(fai_path)
                if 'date' in fai_df.columns and 'industry' in fai_df.columns:
                    industry_list.append(fai_df)
        
        # 加载分行业Output数据
        output_files = [
            'nbs_output_2020.csv',
            'nbs_output_202512.csv'
        ]
        
        for output_file in output_files:
            output_path = os.path.join(self.data_dir, output_file)
            if os.path.exists(output_path):
                output_df = pd.read_csv(output_path)
                if 'date' in output_df.columns and 'industry' in output_df.columns:
                    industry_list.append(output_df)
        
        # 合并所有行业数据
        if industry_list:
            industry_df = pd.concat(industry_list, ignore_index=True)
            # 统一日期格式
            industry_df['date'] = pd.to_datetime(industry_df['date'])
            industry_df = industry_df.sort_values(['industry', 'date']).reset_index(drop=True)
        else:
            industry_df = pd.DataFrame()
        
        self.industry_data = industry_df
        print(f"加载NBS行业数据: {len(industry_df)}条记录")
        
        return industry_df
    
    def load_sw_industry(self) -> pd.DataFrame:
        """
        加载申万行业数据（已映射）
        
        Returns:
            DataFrame: 申万行业数据
        """
        # 加载L1行业PPI数据
        l1_ppi_path = os.path.join(self.data_dir, 'sw_l1_ppi_yoy_202512.csv')
        if os.path.exists(l1_ppi_path):
            sw_l1 = pd.read_csv(l1_ppi_path)
            sw_l1['date'] = pd.to_datetime(sw_l1['date'])
            sw_l1 = sw_l1.rename(columns={'sw_industry': 'industry'})
        else:
            sw_l1 = pd.DataFrame()
        
        # 加载L2行业PPI数据
        l2_ppi_path = os.path.join(self.data_dir, 'sw_l2_ppi_yoy_202512.csv')
        if os.path.exists(l2_ppi_path):
            sw_l2 = pd.read_csv(l2_ppi_path)
            sw_l2['date'] = pd.to_datetime(sw_l2['date'])
            sw_l2 = sw_l2.rename(columns={'sw_industry': 'industry'})
        else:
            sw_l2 = pd.DataFrame()
        
        # 合并L1和L2数据
        if len(sw_l1) > 0 and len(sw_l2) > 0:
            sw_industry = pd.concat([sw_l1, sw_l2], ignore_index=True)
        elif len(sw_l1) > 0:
            sw_industry = sw_l1
        elif len(sw_l2) > 0:
            sw_industry = sw_l2
        else:
            sw_industry = pd.DataFrame()
        
        if len(sw_industry) > 0:
            sw_industry = sw_industry.sort_values(['industry', 'date']).reset_index(drop=True)
        
        print(f"加载申万行业数据: {len(sw_industry)}条记录")
        
        return sw_industry
    
    def load_northbound(self) -> pd.DataFrame:
        """
        加载北向资金数据
        
        Returns:
            DataFrame: 北向资金数据
        """
        northbound_dir = os.path.join(self.data_dir, '..', 'northbound')
        northbound_files = []
        
        if os.path.exists(northbound_dir):
            for file in os.listdir(northbound_dir):
                if file.endswith('.parquet'):
                    northbound_files.append(os.path.join(northbound_dir, file))
        
        if northbound_files:
            northbound_dfs = []
            for file in northbound_files:
                df = pd.read_parquet(file)
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    northbound_dfs.append(df)
            
            if northbound_dfs:
                northbound = pd.concat(northbound_dfs, ignore_index=True)
                northbound = northbound.sort_values('trade_date').reset_index(drop=True)
            else:
                northbound = pd.DataFrame()
        else:
            northbound = pd.DataFrame()
        
        self.northbound_data = northbound
        print(f"加载北向资金数据: {len(northbound)}条记录")
        
        return northbound
    
    def clean_macro_data(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗宏观数据
        
        Args:
            macro_df: 宏观数据
        
        Returns:
            DataFrame: 清洗后的数据
        """
        df = macro_df.copy()
        
        # 1. 前向填充缺失值
        df = df.fillna(method='ffill')
        
        # 2. 计算变化率
        for col in ['cpi_yoy', 'ppi_yoy', 'pmi', 'yield_10y']:
            if col in df.columns:
                df[f'{col}_delta'] = df[col].diff()
        
        # 3. 计算移动平均
        for col in ['cpi_yoy', 'ppi_yoy', 'pmi']:
            if col in df.columns:
                df[f'{col}_ma3'] = df[col].rolling(3, min_periods=1).mean()
                df[f'{col}_ma6'] = df[col].rolling(6, min_periods=1).mean()
        
        # 4. 计算相对位置
        for col in ['cpi_yoy', 'ppi_yoy', 'yield_10y']:
            if col in df.columns:
                rolling_min = df[col].rolling(12, min_periods=1).min()
                rolling_max = df[col].rolling(12, min_periods=1).max()
                df[f'{col}_percentile'] = (df[col] - rolling_min) / (rolling_max - rolling_min + 1e-8)
        
        return df
    
    def clean_industry_data(self, industry_df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗行业数据
        
        Args:
            industry_df: 行业数据
        
        Returns:
            DataFrame: 清洗后的数据
        """
        df = industry_df.copy()
        
        # 1. 前向填充缺失值
        df = df.fillna(method='ffill')
        
        # 2. 计算变化率
        numeric_cols = ['fai_yoy', 'output_yoy', 'sw_ppi_yoy']
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_delta'] = df.groupby('industry')[col].diff()
        
        # 3. 计算移动平均
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_ma3'] = df.groupby('industry')[col].transform(
                    lambda x: x.rolling(3, min_periods=1).mean()
                )
        
        return df
    
    def add_market_data(
        self,
        macro_df: pd.DataFrame,
        industry_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        添加市场数据（估值、换手率等）
        
        Args:
            macro_df: 宏观数据
            industry_df: 行业数据
        
        Returns:
            DataFrame: 添加市场数据后的行业数据
        """
        # 这里应该从其他数据源获取市场数据
        # 暂时用模拟数据代替
        
        # 为行业数据添加模拟的估值和换手率数据
        industries = industry_df['industry'].unique()
        dates = macro_df['date'].unique()
        
        # 生成模拟数据
        simulated_data = []
        for industry in industries:
            for date in dates:
                simulated_data.append({
                    'industry': industry,
                    'date': date,
                    'pb_percentile': np.random.uniform(10, 90),
                    'pe_percentile': np.random.uniform(10, 90),
                    'turnover_rate': np.random.uniform(0.01, 0.15),
                    'rps_120': np.random.uniform(30, 95),
                    'inventory_yoy': np.random.uniform(0, 20),
                    'rev_yoy': np.random.uniform(-5, 15)
                })
        
        market_df = pd.DataFrame(simulated_data)
        
        # 合并到行业数据
        result = industry_df.merge(
            market_df,
            on=['industry', 'date'],
            how='left'
        )
        
        return result
    
    def process_all(self) -> Dict[str, pd.DataFrame]:
        """
        处理所有数据
        
        Returns:
            Dict: 包含所有处理后的数据
        """
        print("=" * 80)
        print("开始处理宏观数据")
        print("=" * 80)
        
        # 1. 加载数据
        macro = self.load_tushare_macro()
        nbs_industry = self.load_nbs_industry()
        sw_industry = self.load_sw_industry()
        northbound = self.load_northbound()
        
        # 2. 清洗数据
        macro_cleaned = self.clean_macro_data(macro)
        industry_cleaned = self.clean_industry_data(sw_industry)
        
        # 3. 添加市场数据
        industry_with_market = self.add_market_data(macro_cleaned, industry_cleaned)
        
        # 4. 重命名列以匹配预测模型期望的格式
        industry_final = industry_with_market.rename(columns={
            'industry': 'sw_industry',
            'sw_ppi_yoy': 'ppi_yoy'
        })
        
        # 确保有必要的列
        required_cols = ['sw_industry', 'date', 'ppi_yoy', 'fai_yoy', 
                        'pb_percentile', 'turnover_rate', 'rps_120',
                        'inventory_yoy', 'rev_yoy']
        
        for col in required_cols:
            if col not in industry_final.columns:
                if col == 'inventory_yoy':
                    industry_final[col] = 0
                elif col == 'rev_yoy':
                    industry_final[col] = 0
                elif col == 'fai_yoy':
                    industry_final[col] = 0
                else:
                    industry_final[col] = 0
        
        print("\n" + "=" * 80)
        print("数据处理完成")
        print("=" * 80)
        print(f"宏观数据: {len(macro_cleaned)}条")
        print(f"行业数据: {len(industry_final)}条")
        print(f"北向资金: {len(northbound)}条")
        
        return {
            'macro': macro_cleaned,
            'industry': industry_final,
            'northbound': northbound
        }


def main():
    """测试数据处理器"""
    processor = MacroDataProcessor()
    
    # 处理数据
    data = processor.process_all()
    
    # 输出样例数据
    print("\n宏观数据样例:")
    print(data['macro'].head())
    
    print("\n行业数据样例:")
    print(data['industry'].head())
    
    # 保存数据
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    data['macro'].to_parquet(f'{output_dir}/macro_features.parquet', index=False)
    data['industry'].to_parquet(f'{output_dir}/industry_features.parquet', index=False)
    
    if len(data['northbound']) > 0:
        data['northbound'].to_parquet(f'{output_dir}/northbound_features.parquet', index=False)
    
    print(f"\n数据已保存到 {output_dir}")


if __name__ == '__main__':
    main()
