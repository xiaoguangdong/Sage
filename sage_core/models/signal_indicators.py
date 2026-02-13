#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
三大核心信号指标计算模块

1. 宏观指标：PMI拐点 + 利率曲线斜率（10Y-2Y利差突破阈值）
2. 行业景气度：营收增速二阶导>0 且 毛利率扩张
3. 资金流：北向资金行业配置比例突破布林带上轨
"""

import pandas as pd
import numpy as np
from pathlib import Path

from scripts.data._shared.runtime import get_data_path

class MacroSignal:
    """宏观信号指标：PMI拐点 + 利率曲线斜率"""
    
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir or str(get_data_path("raw", "tushare", "macro")))
    
    def load_data(self):
        """加载PMI和国债收益率数据"""
        # PMI数据
        pmi_path = self.data_dir / 'tushare_pmi.parquet'
        pmi = pd.read_parquet(pmi_path)
        if 'MONTH' in pmi.columns:
            pmi = pmi[['MONTH', 'PMI010000']].copy()
            pmi.columns = ['month', 'pmi']
        elif 'month' in pmi.columns:
            pmi = pmi[['month', 'PMI010000']].copy()
            pmi.columns = ['month', 'pmi']
        elif 'date' in pmi.columns:
            pmi = pmi[['date', 'PMI010000']].copy()
            pmi.columns = ['month', 'pmi']
            pmi['month'] = pd.to_datetime(pmi['month']).dt.strftime('%Y%m')
        else:
            raise ValueError(f"PMI字段缺失: {pmi_path}")
        pmi['month'] = pmi['month'].astype(str)
        pmi = pmi.sort_values('month').reset_index(drop=True)
        
        # 10年国债收益率
        y10_path = self.data_dir / 'yield_10y.parquet'
        if not y10_path.exists():
            y10_path = self.data_dir / 'tushare_yield_10y.parquet'
        y10 = pd.read_parquet(y10_path)
        if 'yield' in y10.columns:
            y10 = y10.rename(columns={'yield': 'yield_10y'})
        y10['trade_date'] = y10['trade_date'].astype(str)
        
        # 2年国债收益率
        y2_path = self.data_dir / 'yield_2y.parquet'
        if y2_path.exists():
            y2 = pd.read_parquet(y2_path)
            y2 = y2.rename(columns={'yield': 'yield_2y'})
            y2['trade_date'] = y2['trade_date'].astype(str)
        else:
            y2 = None
            print("警告: 2年国债收益率数据不存在")
        
        return pmi, y10, y2
    
    def calculate_pmi_turning_point(self, pmi, window=3, confirm_window=2):
        """
        计算PMI拐点
        
        拐点定义：
        - 上升拐点：PMI从下降转为上升，且连续window个月上升
        - 下降拐点：PMI从上升转为下降，且连续window个月下降
        
        Returns:
            DataFrame: 包含pmi, pmi_ma, pmi_diff, turning_point列
        """
        df = pmi.copy()
        df['pmi_ma'] = df['pmi'].rolling(window=window).mean()
        df['pmi_diff'] = df['pmi'].diff()
        
        # 拐点检测
        df['turning_point'] = 0
        up_confirm = df['pmi_diff'].rolling(confirm_window).sum() > 0
        down_confirm = df['pmi_diff'].rolling(confirm_window).sum() < 0
        df.loc[(df['pmi_diff'].shift(1) < 0) & up_confirm, 'turning_point'] = 1
        df.loc[(df['pmi_diff'].shift(1) > 0) & down_confirm, 'turning_point'] = -1
        
        return df
    
    def calculate_yield_spread(self, y10, y2=None):
        """
        计算利率曲线斜率（10Y-2Y利差）
        
        Args:
            y10: 10年国债收益率DataFrame
            y2: 2年国债收益率DataFrame (可选)
        
        Returns:
            DataFrame: 包含yield_10y, yield_2y, spread列
        """
        df = y10.copy()
        
        if y2 is not None:
            df = df.merge(y2[['trade_date', 'yield_2y']], on='trade_date', how='left')
            df['spread'] = df['yield_10y'] - df['yield_2y']
        else:
            # 如果没有2年期数据，使用代理指标
            df['spread'] = None
        
        return df
    
    def calculate_spread_signal(self, spread_df, threshold=0.5, window=20, mode="threshold"):
        """
        计算利差突破信号
        
        信号定义：利差突破布林带上轨（均线 + N倍标准差）
        
        Args:
            spread_df: 利差数据
            threshold: 布林带倍数
            window: 移动窗口
        
        Returns:
            DataFrame: 包含spread_ma, spread_std, upper_band, signal列
        """
        df = spread_df.copy()
        
        if 'spread' not in df.columns or df['spread'].isna().all():
            df['spread_ma'] = None
            df['spread_std'] = None
            df['upper_band'] = None
            df['lower_band'] = None
            df['spread_signal'] = 0
            return df
        
        df['spread_signal'] = 0

        if mode == "bollinger":
            df['spread_ma'] = df['spread'].rolling(window=window).mean()
            df['spread_std'] = df['spread'].rolling(window=window).std()
            df['upper_band'] = df['spread_ma'] + threshold * df['spread_std']
            df['lower_band'] = df['spread_ma'] - threshold * df['spread_std']
            df.loc[df['spread'] > df['upper_band'], 'spread_signal'] = 1
            df.loc[df['spread'] < df['lower_band'], 'spread_signal'] = -1
        else:
            df['spread_ma'] = None
            df['spread_std'] = None
            df['upper_band'] = threshold
            df['lower_band'] = -threshold
            df.loc[df['spread'] > threshold, 'spread_signal'] = 1
            df.loc[df['spread'] < -threshold, 'spread_signal'] = -1
        
        return df
    
    def get_macro_signal(self, spread_threshold=0.5, spread_window=20, spread_mode="threshold"):
        """
        获取综合宏观信号
        
        Returns:
            dict: 包含pmi_signal和spread_signal
        """
        pmi, y10, y2 = self.load_data()
        
        # PMI拐点信号
        pmi_signal = self.calculate_pmi_turning_point(pmi)
        
        # 利差信号
        spread_df = self.calculate_yield_spread(y10, y2)
        spread_signal = self.calculate_spread_signal(
            spread_df,
            threshold=spread_threshold,
            window=spread_window,
            mode=spread_mode
        )
        
        return {
            'pmi_signal': pmi_signal,
            'spread_signal': spread_signal,
            'pmi_data': pmi,
            'y10_data': y10,
            'y2_data': y2
        }


class IndustryProsperity:
    """行业景气度指标：营收增速二阶导 + 毛利率扩张"""
    
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir or str(get_data_path("raw", "tushare", "fundamental")))
    
    def load_data(self):
        """加载财务数据"""
        all_data = []
        for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
            filepath = self.data_dir / f'fina_indicator_{year}.parquet'
            if filepath.exists():
                df = pd.read_parquet(filepath)
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None
    
    def calculate_revenue_acceleration(self, df, epsilon=0.01):
        """
        计算营收增速二阶导（加速度）
        
        公式：Δ(ΔRevenue/Δt) > ε
        
        Args:
            df: 财务数据，需包含ts_code, end_date, tr_yoy(营收增速)
            epsilon: 阈值
        
        Returns:
            DataFrame: 包含revenue_acceleration, acceleration_signal列
        """
        result = df.copy()
        
        # tr_yoy是营收同比增速
        if 'tr_yoy' not in result.columns:
            print("警告: 缺少tr_yoy字段")
            return result
        
        # 按股票和日期排序
        result = result.sort_values(['ts_code', 'end_date'])
        
        # 计算一阶导（增速的变化）
        result['revenue_diff'] = result.groupby('ts_code')['tr_yoy'].diff()
        
        # 计算二阶导（加速度）
        result['revenue_acceleration'] = result.groupby('ts_code')['revenue_diff'].diff()
        
        # 信号：二阶导 > epsilon
        result['acceleration_signal'] = 0
        result.loc[result['revenue_acceleration'] > epsilon, 'acceleration_signal'] = 1
        
        return result
    
    def calculate_margin_expansion(self, df):
        """
        计算毛利率扩张
        
        毛利率扩张：本期毛利率 > 上期毛利率
        
        Args:
            df: 财务数据，需包含ts_code, end_date, gross_margin
        
        Returns:
            DataFrame: 包含margin_change, margin_signal列
        """
        result = df.copy()
        
        if 'gross_margin' not in result.columns:
            print("警告: 缺少gross_margin字段")
            return result
        
        result = result.sort_values(['ts_code', 'end_date'])
        
        # 毛利率变化
        result['margin_change'] = result.groupby('ts_code')['gross_margin'].diff()
        
        # 信号：毛利率扩张
        result['margin_signal'] = 0
        result.loc[result['margin_change'] > 0, 'margin_signal'] = 1
        
        return result
    
    def get_prosperity_signal(self, epsilon=0.01):
        """
        获取行业景气度综合信号
        
        综合信号 = 加速度信号 AND 毛利率扩张信号
        
        Returns:
            DataFrame: 包含所有计算结果的DataFrame
        """
        df = self.load_data()
        if df is None:
            return None
        
        # 计算加速度
        df = self.calculate_revenue_acceleration(df, epsilon)
        
        # 计算毛利率扩张
        df = self.calculate_margin_expansion(df)
        
        # 综合信号：两个条件同时满足
        df['prosperity_signal'] = 0
        df.loc[(df['acceleration_signal'] == 1) & (df['margin_signal'] == 1), 'prosperity_signal'] = 1
        
        return df


class NorthboundFlow:
    """资金流指标：北向资金行业配置比例突破布林带"""
    
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir or str(get_data_path("raw", "tushare", "northbound")))
    
    def load_data(self):
        """加载北向资金数据"""
        daily_flow = pd.read_parquet(self.data_dir / 'northbound_daily_flow.parquet')
        return daily_flow

    def load_industry_data(self):
        """加载北向资金行业数据"""
        industry_path = self.data_dir / 'industry_northbound_flow.parquet'
        if not industry_path.exists():
            return None
        return pd.read_parquet(industry_path)
    
    def calculate_bollinger_breakout(self, df, window=20, threshold=2.0):
        """
        计算布林带突破信号
        
        信号定义：北向资金净流入突破布林带上轨
        
        Args:
            df: 北向资金数据
            window: 移动窗口
            threshold: 布林带倍数
        
        Returns:
            DataFrame: 包含布林带和信号列
        """
        result = df.copy()
        result['trade_date'] = result['trade_date'].astype(str)
        result = result.sort_values('trade_date')
        
        # 使用north_money（北向资金净流入）
        if 'north_money' not in result.columns:
            print("警告: 缺少north_money字段")
            return result

        result['north_money'] = pd.to_numeric(result['north_money'], errors='coerce')
        
        # 计算布林带
        result['flow_ma'] = result['north_money'].rolling(window=window).mean()
        result['flow_std'] = result['north_money'].rolling(window=window).std()
        result['upper_band'] = result['flow_ma'] + threshold * result['flow_std']
        result['lower_band'] = result['flow_ma'] - threshold * result['flow_std']
        
        # 计算配置比例（相对于均值的偏离程度）
        result['flow_ratio'] = result['north_money'] / result['flow_ma'].abs()
        
        # 信号：突破上轨
        result['flow_signal'] = 0
        result.loc[result['north_money'] > result['upper_band'], 'flow_signal'] = 1
        result.loc[result['north_money'] < result['lower_band'], 'flow_signal'] = -1
        
        return result
    
    def get_flow_signal(self, window=20, threshold=2.0):
        """
        获取资金流信号
        
        Returns:
            DataFrame: 包含布林带和信号的数据
        """
        df = self.load_data()
        return self.calculate_bollinger_breakout(df, window, threshold)

    def calculate_industry_ratio_bollinger(self, df, window=20, threshold=2.0):
        """
        行业配置比例布林带突破

        Args:
            df: 行业北向数据（industry_code/industry_name/trade_date/vol/ratio）
        """
        result = df.copy()
        result['trade_date'] = pd.to_datetime(result['trade_date'])
        result = result.sort_values(['industry_code', 'trade_date'])

        if 'ratio' in result.columns and result['ratio'].notna().any():
            result['industry_ratio'] = pd.to_numeric(result['ratio'], errors='coerce')
        else:
            result['vol'] = pd.to_numeric(result['vol'], errors='coerce')
            total_vol = result.groupby('trade_date')['vol'].transform('sum')
            result['industry_ratio'] = result['vol'] / total_vol.replace(0, np.nan)

        result['ratio_ma'] = result.groupby('industry_code')['industry_ratio'].transform(
            lambda s: s.rolling(window=window).mean()
        )
        result['ratio_std'] = result.groupby('industry_code')['industry_ratio'].transform(
            lambda s: s.rolling(window=window).std()
        )
        result['upper_band'] = result['ratio_ma'] + threshold * result['ratio_std']
        result['lower_band'] = result['ratio_ma'] - threshold * result['ratio_std']

        result['ratio_signal'] = 0
        result.loc[result['industry_ratio'] > result['upper_band'], 'ratio_signal'] = 1
        result.loc[result['industry_ratio'] < result['lower_band'], 'ratio_signal'] = -1

        return result

    def get_industry_flow_signal(self, window=20, threshold=2.0):
        """
        获取行业配置比例信号（若无行业数据则返回None）
        """
        df = self.load_industry_data()
        if df is None or df.empty:
            return None
        return self.calculate_industry_ratio_bollinger(df, window, threshold)


def calculate_all_signals():
    """计算所有信号并输出结果"""
    print("=" * 60)
    print("计算三大核心信号指标")
    print("=" * 60)
    
    # 1. 宏观信号
    print("\n1. 宏观指标：PMI拐点 + 利率曲线斜率")
    macro = MacroSignal()
    macro_result = macro.get_macro_signal(spread_threshold=0.5, spread_mode="threshold")
    
    print("\n  PMI拐点信号（最近12个月）:")
    pmi_sig = macro_result['pmi_signal'].tail(12)
    print(pmi_sig[['month', 'pmi', 'turning_point']])
    
    if macro_result['y2_data'] is not None:
        print("\n  利差信号（最近10天）:")
        spread_sig = macro_result['spread_signal'].tail(10)
        print(spread_sig[['trade_date', 'spread', 'spread_signal']])
    else:
        print("\n  利差信号: 缺少2年国债收益率数据")
    
    # 2. 行业景气度
    print("\n2. 行业景气度：营收增速二阶导 + 毛利率扩张")
    prosperity = IndustryProsperity()
    prosperity_df = prosperity.get_prosperity_signal()
    
    if prosperity_df is not None:
        # 统计信号分布
        signal_count = prosperity_df['prosperity_signal'].value_counts()
        print(f"\n  景气度信号分布:")
        print(f"    看多(1): {signal_count.get(1, 0)} 条")
        print(f"    中性(0): {signal_count.get(0, 0)} 条")
        
        # 显示最近有信号的股票
        recent_signals = prosperity_df[prosperity_df['prosperity_signal'] == 1].tail(10)
        print(f"\n  最近景气度转好的股票:")
        print(recent_signals[['ts_code', 'end_date', 'tr_yoy', 'gross_margin', 'revenue_acceleration']])
    
    # 3. 资金流信号
    print("\n3. 资金流：北向资金布林带突破")
    flow = NorthboundFlow()
    flow_df = flow.get_flow_signal()
    industry_flow_df = flow.get_industry_flow_signal()
    
    print(f"\n  资金流信号分布:")
    signal_count = flow_df['flow_signal'].value_counts()
    print(f"    突破上轨(1): {signal_count.get(1, 0)} 天")
    print(f"    中性(0): {signal_count.get(0, 0)} 天")
    print(f"    突破下轨(-1): {signal_count.get(-1, 0)} 天")
    
    print("\n  最近资金流信号（最后10天）:")
    print(flow_df[['trade_date', 'north_money', 'flow_ma', 'upper_band', 'flow_signal']].tail(10))

    if industry_flow_df is not None:
        print("\n  行业配置比例信号（最近10条）:")
        print(industry_flow_df[['industry_name', 'trade_date', 'industry_ratio', 'ratio_signal']].tail(10))
    
    return {
        'macro': macro_result,
        'prosperity': prosperity_df,
        'flow': flow_df,
        'industry_flow': industry_flow_df
    }


if __name__ == '__main__':
    calculate_all_signals()
