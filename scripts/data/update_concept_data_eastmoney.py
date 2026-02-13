#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
东方财富概念板块数据更新和计算系统
使用纯东方财富接口，数据来源统一，无需聚合计算

接口说明：
- dc_index: 获取东方财富概念板块的日度行情数据（包含涨跌幅、领涨股等）
- dc_member: 获取概念板块成分股列表（按日期）

优势：
1. 数据来源统一（东方财富）
2. 无需聚合计算（已有概念K线）
3. 性能更好（不需要逐个获取成分股）
4. 数据更准确（官方计算的概念表现）

使用说明:
1. 初始化: python update_concept_data_eastmoney.py --init
2. 周度更新: python update_concept_data_eastmoney.py --update
3. 计算表现: python update_concept_data_eastmoney.py --calculate
"""

import pandas as pd
import tushare as ts
import argparse
import time
import os
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/concept_update_eastmoney.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
TS_TOKEN = '2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e'
TIMEOUT = 30  # 超时时间（秒）
DATA_DIR = 'data/tushare/sectors'
BASE_DATE = '20240924'  # 基准日期（2024年9月底）


class EastMoneyConceptManager:
    """东方财富概念数据管理器"""
    
    def __init__(self, token=TS_TOKEN, timeout=TIMEOUT):
        self.pro = ts.pro_api(token)
        self.timeout = timeout
        self.data_dir = DATA_DIR
        self.market = 'EB'  # 东方财富
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def fetch_concept_daily(self, trade_date=None, max_retries=3):
        """获取概念板块日度行情数据"""
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                if trade_date:
                    # 获取指定日期的数据
                    data = self.pro.dc_index(market=self.market, trade_date=trade_date)
                else:
                    # 获取最新日期的数据
                    data = self.pro.dc_index(market=self.market)
                
                elapsed = time.time() - start_time
                
                if elapsed > self.timeout:
                    logger.warning(f"获取概念数据超时 ({elapsed:.1f}s)")
                    return None
                
                logger.info(f"获取到 {len(data)} 个概念的日度数据")
                return data
                
            except Exception as e:
                logger.warning(f"获取概念数据失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"获取概念数据失败")
                    return None
        
        return None
    
    def fetch_concept_history(self, start_date, end_date):
        """获取概念板块历史行情数据"""
        logger.info(f"获取概念历史数据: {start_date} ~ {end_date}")
        
        all_data = []
        current_date = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)
        
        while current_date <= end_date_ts:
            trade_date_str = current_date.strftime('%Y%m%d')
            
            # 获取该日期的数据
            daily_data = self.fetch_concept_daily(trade_date_str)
            
            if daily_data is not None and len(daily_data) > 0:
                all_data.append(daily_data)
                logger.info(f"  {trade_date_str}: {len(daily_data)} 个概念")
            else:
                logger.info(f"  {trade_date_str}: 无数据")
            
            # 移动到下一个交易日（简单处理，跳过周末）
            current_date += timedelta(days=1)
            if current_date.weekday() >= 5:  # 周六或周日
                current_date += timedelta(days=1)
            
            # 避免触发频率限制
            time.sleep(0.5)
        
        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"总计获取 {len(result_df)} 条记录")
            return result_df
        else:
            logger.error("未能获取任何历史数据")
            return None
    
    def fetch_concept_members(self, concept_code, trade_date, max_retries=3):
        """获取概念成分股列表"""
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                members = self.pro.dc_member(
                    market=self.market,
                    concept_code=concept_code,
                    trade_date=trade_date
                )
                elapsed = time.time() - start_time
                
                if elapsed > self.timeout:
                    logger.warning(f"获取成分股超时: {concept_code} ({elapsed:.1f}s)")
                    return None
                
                return members
                
            except Exception as e:
                logger.warning(f"获取成分股失败: {concept_code} (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    return None
        
        return None
    
    def save_concept_data(self, data, suffix=''):
        """保存概念数据"""
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f'concept_daily_eastmoney_{timestamp}{suffix}.csv'
        filepath = os.path.join(self.data_dir, filename)
        
        data.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"数据已保存: {filepath}")
        
        # 同时保存为最新版本
        latest_path = os.path.join(self.data_dir, f'concept_daily_eastmoney{suffix}.csv')
        data.to_csv(latest_path, index=False, encoding='utf-8-sig')
        logger.info(f"数据已更新: {latest_path}")
        
        return filepath
    
    def load_concept_data(self, filename='concept_daily_eastmoney.csv'):
        """加载概念数据"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"文件不存在: {filepath}")
            return None
        
        data = pd.read_csv(filepath)
        data['trade_date'] = pd.to_datetime(data['trade_date'])
        logger.info(f"加载数据: {len(data)} 条记录")
        return data
    
    def calculate_concept_performance(self, data, min_stock_count=0):
        """计算概念表现"""
        logger.info("计算概念表现...")
        
        # 东方财富数据已经有pct_change，直接使用
        # 按概念分组计算统计指标
        concept_stats = data.groupby('ts_code').agg({
            'name': 'first',
            'pct_change': ['mean', 'std'],
            'turnover_rate': 'mean',
            'up_num': 'sum',
            'down_num': 'sum',
            'total_mv': 'mean'
        }).reset_index()
        
        concept_stats.columns = ['ts_code', 'name', 'avg_return', 'volatility', 'avg_turnover', 'total_up', 'total_down', 'avg_mv']
        
        # 计算累计涨幅（使用pct_change的简单累加，实际应该用复利）
        concept_cumulative = data.groupby('ts_code')['pct_change'].apply(
            lambda x: (1 + x / 100).prod() - 1
        ).reset_index()
        concept_cumulative.columns = ['ts_code', 'total_return']
        
        # 合并数据
        concept_perf = pd.merge(concept_stats, concept_cumulative, on='ts_code')
        
        # 计算上涨天数比例
        total_days = data.groupby('ts_code').size().reset_index(name='total_days')
        concept_perf = pd.merge(concept_perf, total_days, on='ts_code')
        concept_perf['positive_ratio'] = concept_perf['total_up'] / concept_perf['total_days']
        
        # 计算最大回撤
        concept_max_dd = data.groupby('ts_code').apply(
            lambda x: ((1 + x['pct_change'] / 100).cumprod() / (1 + x['pct_change'] / 100).cumprod().cummax() - 1).min() * 100
        ).reset_index()
        concept_max_dd.columns = ['ts_code', 'max_drawdown']
        concept_perf = pd.merge(concept_perf, concept_max_dd, on='ts_code')
        
        # 转换单位
        concept_perf['total_return'] = concept_perf['total_return'] * 100
        concept_perf['avg_return'] = concept_perf['avg_return']
        
        concept_perf = concept_perf.sort_values('total_return', ascending=False)
        
        logger.info(f"计算完成: {len(concept_perf)} 个概念")
        
        return concept_perf
    
    def calculate_score_ranking(self, concept_perf):
        """计算评分排名"""
        logger.info("计算评分排名...")
        
        # 标准化指标
        def normalize_score(series, reverse=False):
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series([50] * len(series), index=series.index)
            if reverse:
                return (series - min_val) / (max_val - min_val) * 100
            else:
                return (max_val - series) / (max_val - min_val) * 100
        
        concept_perf['return_score'] = normalize_score(concept_perf['total_return'], reverse=True)
        concept_perf['drawdown_score'] = normalize_score(concept_perf['max_drawdown'], reverse=False)
        concept_perf['volatility_score'] = normalize_score(concept_perf['volatility'], reverse=False)
        concept_perf['persistence_score'] = concept_perf['positive_ratio'] * 100
        concept_perf['turnover_score'] = normalize_score(concept_perf['avg_turnover'], reverse=True)
        
        # 综合评分
        weights = {
            'return_score': 0.30,
            'drawdown_score': 0.15,
            'volatility_score': 0.10,
            'persistence_score': 0.20,
            'turnover_score': 0.25
        }
        
        concept_perf['total_score'] = (
            concept_perf['return_score'] * weights['return_score'] +
            concept_perf['drawdown_score'] * weights['drawdown_score'] +
            concept_perf['volatility_score'] * weights['volatility_score'] +
            concept_perf['persistence_score'] * weights['persistence_score'] +
            concept_perf['turnover_score'] * weights['turnover_score']
        )
        
        concept_perf = concept_perf.sort_values('total_score', ascending=False)
        
        # 等级划分
        concept_perf['grade'] = pd.cut(concept_perf['total_score'],
                                     bins=[0, 40, 60, 80, 100],
                                     labels=['D', 'C', 'B', 'A'])
        
        logger.info(f"评分计算完成")
        
        return concept_perf


def main():
    parser = argparse.ArgumentParser(description='东方财富概念板块数据更新和计算系统')
    parser.add_argument('--init', action='store_true', help='初始化：获取基准数据')
    parser.add_argument('--update', action='store_true', help='周度更新：获取最新数据')
    parser.add_argument('--calculate', action='store_true', help='计算概念表现')
    parser.add_argument('--start-date', type=str, default='2024-09-24', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='结束日期')
    parser.add_argument('--trade-date', type=str, default=None, help='交易日期（YYYYMMDD）')
    
    args = parser.parse_args()
    
    manager = EastMoneyConceptManager()
    
    if args.init:
        logger.info("=" * 80)
        logger.info("初始化：获取东方财富概念板块基准数据")
        logger.info("=" * 80)
        
        # 获取基准日期的数据
        start_date = pd.Timestamp(args.start_date).strftime('%Y%m%d')
        end_date = pd.Timestamp(args.end_date).strftime('%Y%m%d')
        
        concept_data = manager.fetch_concept_history(start_date, end_date)
        
        if concept_data is not None:
            manager.save_concept_data(concept_data, suffix='_base')
            
            # 计算并保存表现
            concept_perf = manager.calculate_concept_performance(concept_data)
            concept_scored = manager.calculate_score_ranking(concept_perf)
            
            timestamp = datetime.now().strftime('%Y%m%d')
            perf_path = os.path.join(manager.data_dir, f'concept_performance_eastmoney_{timestamp}.csv')
            concept_scored.to_csv(perf_path, index=False, encoding='utf-8-sig')
            logger.info(f"概念表现已保存: {perf_path}")
            
            # 打印Top 10
            print("\n" + "=" * 100)
            print(f"综合评分 Top 10:")
            print("=" * 100)
            print(f"{'排名':<4} {'概念名称':<20} {'综合评分':<10} {'涨幅分':<8} {'回撤分':<8} {'持续分':<8} {'换手分':<8}")
            print("-" * 100)
            for i, row in concept_scored.head(10).iterrows():
                print(f"{i+1:<4} {row['name']:<20} {row['total_score']:>8.1f} {row['return_score']:>7.1f} {row['drawdown_score']:>7.1f} {row['persistence_score']:>7.1f} {row['turnover_score']:>7.1f}")
    
    elif args.update:
        logger.info("=" * 80)
        logger.info("周度更新：获取最新东方财富概念板块数据")
        logger.info("=" * 80)
        
        # 获取最新日期的数据
        trade_date = args.trade_date if args.trade_date else datetime.now().strftime('%Y%m%d')
        
        # 获取最近一个月的数据
        start_date = (pd.Timestamp(trade_date) - timedelta(days=30)).strftime('%Y%m%d')
        
        concept_data = manager.fetch_concept_history(start_date, trade_date)
        
        if concept_data is not None:
            manager.save_concept_data(concept_data)
            
            # 计算并保存表现
            concept_perf = manager.calculate_concept_performance(concept_data)
            concept_scored = manager.calculate_score_ranking(concept_perf)
            
            timestamp = datetime.now().strftime('%Y%m%d')
            perf_path = os.path.join(manager.data_dir, f'concept_performance_eastmoney_{timestamp}.csv')
            concept_scored.to_csv(perf_path, index=False, encoding='utf-8-sig')
            logger.info(f"概念表现已保存: {perf_path}")
            
            # 打印Top 10
            print("\n" + "=" * 100)
            print(f"综合评分 Top 10:")
            print("=" * 100)
            print(f"{'排名':<4} {'概念名称':<20} {'综合评分':<10} {'涨幅分':<8} {'回撤分':<8} {'持续分':<8} {'换手分':<8}")
            print("-" * 100)
            for i, row in concept_scored.head(10).iterrows():
                print(f"{i+1:<4} {row['name']:<20} {row['total_score']:>8.1f} {row['return_score']:>7.1f} {row['drawdown_score']:>7.1f} {row['persistence_score']:>7.1f} {row['turnover_score']:>7.1f}")
    
    elif args.calculate:
        logger.info("=" * 80)
        logger.info("计算概念表现")
        logger.info("=" * 80)
        
        # 加载概念数据
        concept_data = manager.load_concept_data()
        
        if concept_data is not None:
            # 计算并保存表现
            concept_perf = manager.calculate_concept_performance(concept_data)
            concept_scored = manager.calculate_score_ranking(concept_perf)
            
            timestamp = datetime.now().strftime('%Y%m%d')
            perf_path = os.path.join(manager.data_dir, f'concept_performance_eastmoney_{timestamp}.csv')
            concept_scored.to_csv(perf_path, index=False, encoding='utf-8-sig')
            logger.info(f"概念表现已保存: {perf_path}")
            
            # 打印Top 10
            print("\n" + "=" * 100)
            print(f"综合评分 Top 10:")
            print("=" * 100)
            print(f"{'排名':<4} {'概念名称':<20} {'综合评分':<10} {'涨幅分':<8} {'回撤分':<8} {'持续分':<8} {'换手分':<8}")
            print("-" * 100)
            for i, row in concept_scored.head(10).iterrows():
                print(f"{i+1:<4} {row['name']:<20} {row['total_score']:>8.1f} {row['return_score']:>7.1f} {row['drawdown_score']:>7.1f} {row['persistence_score']:>7.1f} {row['turnover_score']:>7.1f}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()