#!/usr/bin/env python3
"""
股票因子计算和横截面排名

计算四大类因子：
- Momentum（动量）：4w_ret, 12w_ret
- Quality（质量）：roe_ttm, gross_margin
- Liquidity（流动性）：turnover, amt_rank
- Risk（风险）：vol_12w, beta

输出每只股票的横截面强弱分（0-100）
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class StockFactorCalculator:
    """股票因子计算器"""
    
    def __init__(self):
        self.data_dir = "data/tushare"
        self.output_dir = "data/tushare/factors"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("股票因子计算器初始化")
    
    def load_data(self):
        """加载所有必要的数据"""
        logger.info("=" * 70)
        logger.info("加载数据...")
        logger.info("=" * 70)
        
        # 加载日线数据（2020-2026）
        logger.info("加载日线数据...")
        daily_data = []
        for year in range(2020, 2027):
            file_path = os.path.join(self.data_dir, "daily", f"daily_{year}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                daily_data.append(df)
                logger.info(f"  {year}年: {len(df)} 条记录")
        
        if daily_data:
            self.daily = pd.concat(daily_data, ignore_index=True)
            self.daily['trade_date'] = pd.to_datetime(self.daily['trade_date'])
            logger.info(f"✓ 日线数据总计: {len(self.daily)} 条记录，{self.daily['ts_code'].nunique()} 只股票")
        else:
            logger.error("✗ 未找到日线数据")
            return False
        
        # 加载daily_basic数据
        logger.info("加载daily_basic数据...")
        basic_file = os.path.join(self.data_dir, "daily_basic_all.parquet")
        if os.path.exists(basic_file):
            self.basic = pd.read_parquet(basic_file)
            self.basic['trade_date'] = pd.to_datetime(self.basic['trade_date'])
            logger.info(f"✓ Daily basic: {len(self.basic)} 条记录")
        else:
            logger.error("✗ 未找到daily_basic数据")
            return False
        
        # 加载财务数据
        logger.info("加载财务数据...")
        fina_data = []
        for year in range(2020, 2026):
            file_path = os.path.join(self.data_dir, "fundamental", f"fina_indicator_{year}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                fina_data.append(df)
                logger.info(f"  {year}年: {len(df)} 条记录")
        
        if fina_data:
            self.fina = pd.concat(fina_data, ignore_index=True)
            self.fina['end_date'] = pd.to_datetime(self.fina['end_date'])
            logger.info(f"✓ 财务数据总计: {len(self.fina)} 条记录，{self.fina['ts_code'].nunique()} 只股票")
        else:
            logger.error("✗ 未找到财务数据")
            return False
        
        # 加载指数数据（用于计算beta）
        logger.info("加载指数数据...")
        index_file = os.path.join(self.data_dir, "index_ohlc_all.parquet")
        if os.path.exists(index_file):
            self.index_data = pd.read_parquet(index_file)
            self.index_data['trade_date'] = pd.to_datetime(self.index_data['trade_date'])
            logger.info(f"✓ 指数数据: {len(self.index_data)} 条记录")
        else:
            logger.error("✗ 未找到指数数据")
            return False
        
        return True
    
    def calculate_momentum_factors(self):
        """计算动量因子"""
        logger.info("\n" + "=" * 70)
        logger.info("计算动量因子（Momentum）...")
        logger.info("=" * 70)
        
        # 按股票分组
        results = []
        
        for ts_code, group in self.daily.groupby('ts_code'):
            group = group.sort_values('trade_date').copy()
            
            # 计算4周（20个交易日）收益率
            for i in range(len(group)):
                if i >= 20:  # 至少需要20天数据
                    current_price = group.iloc[i]['close']
                    price_20d_ago = group.iloc[i-20]['close']
                    ret_4w = (current_price / price_20d_ago - 1) * 100  # 百分比
                    
                    results.append({
                        'ts_code': ts_code,
                        'trade_date': group.iloc[i]['trade_date'],
                        '4w_ret': ret_4w
                    })
            
            # 计算12周（60个交易日）收益率
            for i in range(len(group)):
                if i >= 60:  # 至少需要60天数据
                    current_price = group.iloc[i]['close']
                    price_60d_ago = group.iloc[i-60]['close']
                    ret_12w = (current_price / price_60d_ago - 1) * 100  # 百分比
                    
                    # 更新或添加记录
                    found = False
                    for r in results:
                        if r['ts_code'] == ts_code and r['trade_date'] == group.iloc[i]['trade_date']:
                            r['12w_ret'] = ret_12w
                            found = True
                            break
                    if not found:
                        results.append({
                            'ts_code': ts_code,
                            'trade_date': group.iloc[i]['trade_date'],
                            '4w_ret': np.nan,
                            '12w_ret': ret_12w
                        })
        
        self.momentum_factors = pd.DataFrame(results)
        logger.info(f"✓ 动量因子计算完成: {len(self.momentum_factors)} 条记录")
        
        return self.momentum_factors
    
    def calculate_quality_factors(self):
        """计算质量因子"""
        logger.info("\n" + "=" * 70)
        logger.info("计算质量因子（Quality）...")
        logger.info("=" * 70)
        
        # 财务数据按季度发布，需要将季度数据映射到每日
        results = []
        
        # 获取所有交易日期
        all_dates = sorted(self.daily['trade_date'].unique())
        
        for ts_code in self.fina['ts_code'].unique():
            stock_fina = self.fina[self.fina['ts_code'] == ts_code].sort_values('end_date')
            
            for trade_date in all_dates:
                # 找到该交易日最近的财务报告
                available_reports = stock_fina[stock_fina['end_date'] <= trade_date]
                
                if not available_reports.empty:
                    # 使用最新的财务报告
                    latest_report = available_reports.iloc[-1]
                    
                    results.append({
                        'ts_code': ts_code,
                        'trade_date': trade_date,
                        'roe_ttm': latest_report.get('roe', np.nan),
                        'gross_margin': latest_report.get('gross_margin', np.nan)
                    })
        
        self.quality_factors = pd.DataFrame(results)
        logger.info(f"✓ 质量因子计算完成: {len(self.quality_factors)} 条记录")
        
        return self.quality_factors
    
    def calculate_liquidity_factors(self):
        """计算流动性因子"""
        logger.info("\n" + "=" * 70)
        logger.info("计算流动性因子（Liquidity）...")
        logger.info("=" * 70)
        
        results = []
        
        # 按日期分组计算成交额排名
        for trade_date, group in self.daily.groupby('trade_date'):
            # 计算成交额排名
            group_sorted = group.sort_values('amount', ascending=False)
            group_sorted['amt_rank'] = range(1, len(group_sorted) + 1)
            
            for _, row in group_sorted.iterrows():
                results.append({
                    'ts_code': row['ts_code'],
                    'trade_date': trade_date,
                    'turnover': np.nan,  # 稍后从daily_basic获取
                    'amt_rank': row['amt_rank']
                })
        
        self.liquidity_factors = pd.DataFrame(results)
        
        # 从daily_basic获取换手率
        for _, row in self.basic.iterrows():
            mask = (self.liquidity_factors['ts_code'] == row['ts_code']) & \
                   (self.liquidity_factors['trade_date'] == row['trade_date'])
            self.liquidity_factors.loc[mask, 'turnover'] = row['turnover_rate']
        
        logger.info(f"✓ 流动性因子计算完成: {len(self.liquidity_factors)} 条记录")
        
        return self.liquidity_factors
    
    def calculate_risk_factors(self):
        """计算风险因子"""
        logger.info("\n" + "=" * 70)
        logger.info("计算风险因子（Risk）...")
        logger.info("=" * 70)
        
        results = []
        
        # 获取沪深300指数
        hs300_data = self.index_data[self.index_data['ts_code'] == '000300.SH'].copy()
        hs300_data = hs300_data.sort_values('trade_date')
        
        for ts_code, group in self.daily.groupby('ts_code'):
            group = group.sort_values('trade_date').copy()
            
            # 合并指数数据
            merged = pd.merge(
                group[['trade_date', 'close']],
                hs300_data[['trade_date', 'close']].rename(columns={'close': 'index_close'}),
                on='trade_date',
                how='inner'
            )
            
            # 计算每日收益率
            merged['stock_ret'] = merged['close'].pct_change() * 100
            merged['index_ret'] = merged['index_close'].pct_change() * 100
            
            for i in range(len(merged)):
                if i >= 60:  # 至少需要60天数据计算波动率
                    # 计算12周波动率
                    rets_60d = merged['stock_ret'].iloc[i-60:i]
                    vol_12w = rets_60d.std()
                    
                    # 计算beta（使用60天数据）
                    stock_rets = merged['stock_ret'].iloc[i-60:i]
                    index_rets = merged['index_ret'].iloc[i-60:i]
                    
                    # beta = Cov(stock, market) / Var(market)
                    if len(stock_rets) > 1 and len(index_rets) > 1:
                        covariance = np.cov(stock_rets, index_rets)[0, 1]
                        variance = np.var(index_rets)
                        beta = covariance / variance if variance != 0 else np.nan
                    else:
                        beta = np.nan
                    
                    results.append({
                        'ts_code': ts_code,
                        'trade_date': merged.iloc[i]['trade_date'],
                        'vol_12w': vol_12w,
                        'beta': beta
                    })
        
        self.risk_factors = pd.DataFrame(results)
        logger.info(f"✓ 风险因子计算完成: {len(self.risk_factors)} 条记录")
        
        return self.risk_factors
    
    def merge_all_factors(self):
        """合并所有因子"""
        logger.info("\n" + "=" * 70)
        logger.info("合并所有因子...")
        logger.info("=" * 70)
        
        # 从动量因子开始
        df = self.momentum_factors.copy()
        
        # 合并质量因子
        df = pd.merge(df, self.quality_factors, on=['ts_code', 'trade_date'], how='left')
        
        # 合并流动性因子
        df = pd.merge(df, self.liquidity_factors, on=['ts_code', 'trade_date'], how='left')
        
        # 合并风险因子
        df = pd.merge(df, self.risk_factors, on=['ts_code', 'trade_date'], how='left')
        
        logger.info(f"✓ 因子合并完成: {len(df)} 条记录")
        
        return df
    
    def calculate_rank_scores(self, df):
        """计算横截面排名得分（0-100）"""
        logger.info("\n" + "=" * 70)
        logger.info("计算横截面排名得分...")
        logger.info("=" * 70)
        
        results = []
        
        # 按日期分组
        for trade_date, group in df.groupby('trade_date'):
            # 计算每个因子的排名（0-100）
            # 排名规则：值越大越好
            
            # 4w_ret：越大越好
            if '4w_ret' in group.columns:
                group['rank_4w_ret'] = group['4w_ret'].rank(pct=True) * 100
            
            # 12w_ret：越大越好
            if '12w_ret' in group.columns:
                group['rank_12w_ret'] = group['12w_ret'].rank(pct=True) * 100
            
            # roe_ttm：越大越好
            if 'roe_ttm' in group.columns:
                group['rank_roe_ttm'] = group['roe_ttm'].rank(pct=True) * 100
            
            # gross_margin：越大越好
            if 'gross_margin' in group.columns:
                group['rank_gross_margin'] = group['gross_margin'].rank(pct=True) * 100
            
            # turnover：适中为好，使用绝对值距离中位数的排名
            if 'turnover' in group.columns:
                median_turnover = group['turnover'].median()
                group['rank_turnover'] = 100 - abs(group['turnover'] - median_turnover).rank(pct=True) * 100
            
            # amt_rank：越小越好（排名越靠前越好）
            if 'amt_rank' in group.columns:
                group['rank_amt_rank'] = 100 - group['amt_rank'].rank(pct=True) * 100
            
            # vol_12w：越小越好
            if 'vol_12w' in group.columns:
                group['rank_vol_12w'] = 100 - group['vol_12w'].rank(pct=True) * 100
            
            # beta：适中为好（接近1为好）
            if 'beta' in group.columns:
                group['rank_beta'] = 100 - abs(group['beta'] - 1).rank(pct=True) * 100
            
            # 计算综合得分
            rank_cols = [col for col in group.columns if col.startswith('rank_')]
            if rank_cols:
                group['score'] = group[rank_cols].mean(axis=1)
            
            results.append(group)
        
        result_df = pd.concat(results, ignore_index=True)
        logger.info(f"✓ 排名得分计算完成: {len(result_df)} 条记录")
        
        return result_df
    
    def save_results(self, df):
        """保存结果"""
        logger.info("\n" + "=" * 70)
        logger.info("保存结果...")
        logger.info("=" * 70)
        
        # 保存完整因子数据
        output_file = os.path.join(self.output_dir, "stock_factors_with_score.parquet")
        df.to_parquet(output_file)
        logger.info(f"✓ 完整因子数据: {output_file}")
        
        # 保存最近一天的得分
        latest_date = df['trade_date'].max()
        latest_df = df[df['trade_date'] == latest_date].copy()
        latest_df = latest_df.sort_values('score', ascending=False)
        
        latest_file = os.path.join(self.output_dir, f"latest_scores_{latest_date.strftime('%Y%m%d')}.csv")
        latest_df.to_csv(latest_file, index=False)
        logger.info(f"✓ 最新得分: {latest_file}")
        logger.info(f"  日期: {latest_date.strftime('%Y-%m-%d')}")
        logger.info(f"  股票数: {len(latest_df)}")
        
        return latest_df
    
    def run(self):
        """执行完整的因子计算流程"""
        logger.info("\n" + "=" * 70)
        logger.info("开始计算股票因子...")
        logger.info("=" * 70)
        
        # 1. 加载数据
        if not self.load_data():
            logger.error("数据加载失败")
            return None
        
        # 2. 计算各类因子
        self.calculate_momentum_factors()
        self.calculate_quality_factors()
        self.calculate_liquidity_factors()
        self.calculate_risk_factors()
        
        # 3. 合并所有因子
        df = self.merge_all_factors()
        
        # 4. 计算排名得分
        df = self.calculate_rank_scores(df)
        
        # 5. 保存结果
        latest_df = self.save_results(df)
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ 因子计算完成！")
        logger.info("=" * 70)
        
        return df, latest_df


def main():
    """主函数"""
    calculator = StockFactorCalculator()
    df, latest_df = calculator.run()
    
    if df is not None and latest_df is not None:
        print("\n" + "=" * 70)
        print("最新得分Top 20股票：")
        print("=" * 70)
        print(latest_df[['ts_code', 'score', '4w_ret', '12w_ret', 'roe_ttm', 'gross_margin', 'turnover', 'amt_rank', 'vol_12w', 'beta']].head(20).to_string())


if __name__ == "__main__":
    main()