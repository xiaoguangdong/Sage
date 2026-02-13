#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
概念成分股数据更新和计算系统
每周更新概念列表和成分股，计算概念表现

使用说明:
1. 初始化: python update_concept_data.py --init  # 获取基准数据（如2024-09）
2. 周度更新: python update_concept_data.py --update  # 获取最新数据并计算
3. 计算表现: python update_concept_data.py --calculate  # 只计算概念表现
"""

import pandas as pd
import tushare as ts
import argparse
import time
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path

from scripts.data._shared.runtime import get_tushare_token, setup_logger

logger = setup_logger(Path(__file__).stem)

# 配置参数
TS_TOKEN = get_tushare_token()
TIMEOUT = 30  # 超时时间（秒）
DATA_DIR = 'data/tushare/sectors'
BASE_DATE = '20240930'  # 基准日期（2024年9月底）


class ConceptDataManager:
    """概念数据管理器"""
    
    def __init__(self, token=TS_TOKEN, timeout=TIMEOUT):
        self.pro = ts.pro_api(token)
        self.timeout = timeout
        self.data_dir = DATA_DIR
        
        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def fetch_concept_list(self):
        """获取概念列表"""
        logger.info("获取概念列表...")
        concepts = self.pro.concept()
        logger.info(f"获取到 {len(concepts)} 个概念")
        return concepts
    
    def fetch_concept_members(self, concept_id, max_retries=3):
        """获取概念成分股（带超时和重试）"""
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                members = self.pro.concept_detail(id=concept_id)
                elapsed = time.time() - start_time
                
                if elapsed > self.timeout:
                    logger.warning(f"概念 {concept_id} 获取超时 ({elapsed:.1f}s)")
                    return None
                
                return members
                
            except Exception as e:
                logger.warning(f"概念 {concept_id} 获取失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"概念 {concept_id} 获取失败，跳过")
                    return None
        
        return None
    
    def fetch_all_concept_members(self, concepts=None):
        """获取所有概念的成分股"""
        if concepts is None:
            concepts = self.fetch_concept_list()
        
        logger.info(f"开始获取 {len(concepts)} 个概念的成分股...")
        
        all_members = []
        failed_concepts = []
        timeout_concepts = []
        
        for idx, row in concepts.iterrows():
            concept_id = row['code']
            concept_name = row['name']
            
            members = self.fetch_concept_members(concept_id)
            
            if members is not None and len(members) > 0:
                members['concept_id'] = concept_id
                members['concept_name'] = concept_name
                members['fetch_date'] = datetime.now().strftime('%Y-%m-%d')
                all_members.append(members)
                logger.info(f"[{idx+1:3d}/{len(concepts)}] {concept_name:<20} 成分股: {len(members):3d} 只")
            elif members is None:
                timeout_concepts.append((concept_id, concept_name))
            else:
                failed_concepts.append((concept_id, concept_name))
            
            # 避免触发频率限制
            time.sleep(0.3)
        
        # 合并结果
        if all_members:
            result_df = pd.concat(all_members, ignore_index=True)
            logger.info(f"\n成功获取 {len(result_df)} 条成分股记录")
            logger.info(f"成功概念数: {result_df['concept_name'].nunique()}")
            logger.info(f"失败概念数: {len(failed_concepts)}")
            logger.info(f"超时概念数: {len(timeout_concepts)}")
            
            return result_df, failed_concepts, timeout_concepts
        else:
            logger.error("未能获取任何成分股数据")
            return None, failed_concepts, timeout_concepts
    
    def save_concept_data(self, data, suffix=''):
        """保存概念数据"""
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f'all_concept_details_{timestamp}{suffix}.csv'
        filepath = os.path.join(self.data_dir, filename)
        
        data.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"数据已保存: {filepath}")
        
        # 同时保存为最新版本
        latest_path = os.path.join(self.data_dir, f'all_concept_details{suffix}.csv')
        data.to_csv(latest_path, index=False, encoding='utf-8-sig')
        logger.info(f"数据已更新: {latest_path}")
        
        return filepath
    
    def load_concept_data(self, filename='all_concept_details.csv'):
        """加载概念数据"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.error(f"文件不存在: {filepath}")
            return None
        
        data = pd.read_csv(filepath)
        logger.info(f"加载数据: {len(data)} 条记录")
        return data
    
    def load_stock_data(self, start_date, end_date):
        """加载个股数据"""
        # 根据日期范围加载对应的parquet文件
        year = start_date[:4]
        filepath = f'data/tushare/daily/daily_{year}.parquet'
        
        if not os.path.exists(filepath):
            logger.error(f"个股数据文件不存在: {filepath}")
            return None
        
        daily_data = pd.read_parquet(filepath)
        daily_data['trade_date'] = pd.to_datetime(daily_data['trade_date'])
        
        # 筛选日期范围
        period_data = daily_data[
            (daily_data['trade_date'] >= pd.Timestamp(start_date)) &
            (daily_data['trade_date'] <= pd.Timestamp(end_date))
        ]
        
        logger.info(f"加载个股数据: {len(period_data)} 条记录")
        return period_data
    
    def calculate_concept_performance(self, concept_data, stock_data, min_stock_count=10):
        """计算概念表现"""
        logger.info("计算概念表现...")
        
        # 筛选成分股数>=10的概念
        stock_count = concept_data.groupby('concept_name')['ts_code'].count()
        valid_concepts = stock_count[stock_count >= min_stock_count].index.tolist()
        concept_filtered = concept_data[concept_data['concept_name'].isin(valid_concepts)]
        
        logger.info(f"有效概念数（成分股>={min_stock_count}）: {len(valid_concepts)}")
        
        # 计算每个概念的表现
        concept_performance = []
        concept_weekly_data = {}
        
        # 添加周度标识
        stock_data['week_start'] = stock_data['trade_date'] - pd.to_timedelta(stock_data['trade_date'].dt.dayofweek, unit='D')
        stock_data['year_week'] = stock_data['week_start'].dt.strftime('%Y-%U')
        
        for concept_name in concept_filtered['concept_name'].unique():
            stocks = concept_filtered[concept_filtered['concept_name'] == concept_name]['ts_code'].tolist()
            concept_stocks = stock_data[stock_data['ts_code'].isin(stocks)]
            
            if len(concept_stocks) > 0:
                # 日度表现
                concept_daily = concept_stocks.groupby('trade_date')['pct_chg'].mean().reset_index()
                concept_daily.columns = ['trade_date', 'concept_return']
                concept_cumulative = (1 + concept_daily['concept_return'] / 100).cumprod() - 1
                
                total_return = concept_cumulative.iloc[-1] * 100
                avg_daily_return = concept_daily['concept_return'].mean()
                max_drawdown = (concept_cumulative / concept_cumulative.cummax() - 1).min() * 100
                volatility = concept_daily['concept_return'].std()
                
                # 周度表现
                concept_stocks['year_week'] = concept_stocks['week_start'].dt.strftime('%Y-%U')
                weekly_return = concept_stocks.groupby('year_week')['pct_chg'].mean()
                concept_weekly_data[concept_name] = weekly_return
                
                positive_weeks = (weekly_return > 0).sum()
                total_weeks = len(weekly_return)
                
                concept_performance.append({
                    'concept_name': concept_name,
                    'stock_count': len(stocks),
                    'total_return': total_return,
                    'avg_daily_return': avg_daily_return,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'positive_weeks': positive_weeks,
                    'total_weeks': total_weeks,
                    'positive_week_ratio': positive_weeks / total_weeks if total_weeks > 0 else 0
                })
        
        concept_df = pd.DataFrame(concept_performance)
        concept_df = concept_df.sort_values('total_return', ascending=False)
        
        logger.info(f"计算完成: {len(concept_df)} 个概念")
        
        return concept_df, concept_weekly_data
    
    def calculate_score_ranking(self, concept_df, concept_weekly_data):
        """计算评分排名"""
        logger.info("计算评分排名...")
        
        # 计算周度上榜次数
        weekly_df = pd.DataFrame(concept_weekly_data).T
        weekly_top_counts = {}
        
        for week in sorted(weekly_df.columns):
            top_5 = weekly_df[week].sort_values(ascending=False).head(5).index.tolist()
            for concept in top_5:
                weekly_top_counts[concept] = weekly_top_counts.get(concept, 0) + 1
        
        concept_df['top_count'] = concept_df['concept_name'].map(weekly_top_counts).fillna(0)
        concept_df['top_ratio'] = concept_df['top_count'] / len(sorted(weekly_df.columns))
        
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
        
        concept_df['return_score'] = normalize_score(concept_df['total_return'], reverse=True)
        concept_df['drawdown_score'] = normalize_score(concept_df['max_drawdown'], reverse=False)
        concept_df['volatility_score'] = normalize_score(concept_df['volatility'], reverse=False)
        concept_df['persistence_score'] = concept_df['positive_week_ratio'] * 100
        concept_df['top_score'] = concept_df['top_ratio'] * 100
        concept_df['stock_count_score'] = (concept_df['stock_count'] / 50 * 100).clip(upper=100)
        
        # 综合评分
        weights = {
            'return_score': 0.25,
            'drawdown_score': 0.15,
            'volatility_score': 0.10,
            'persistence_score': 0.20,
            'top_score': 0.20,
            'stock_count_score': 0.10
        }
        
        concept_df['total_score'] = (
            concept_df['return_score'] * weights['return_score'] +
            concept_df['drawdown_score'] * weights['drawdown_score'] +
            concept_df['volatility_score'] * weights['volatility_score'] +
            concept_df['persistence_score'] * weights['persistence_score'] +
            concept_df['top_score'] * weights['top_score'] +
            concept_df['stock_count_score'] * weights['stock_count_score']
        )
        
        concept_df = concept_df.sort_values('total_score', ascending=False)
        
        # 等级划分
        concept_df['grade'] = pd.cut(concept_df['total_score'],
                                     bins=[0, 40, 60, 80, 100],
                                     labels=['D', 'C', 'B', 'A'])
        
        logger.info(f"评分计算完成")
        
        return concept_df


def main():
    parser = argparse.ArgumentParser(description='概念成分股数据更新和计算系统')
    parser.add_argument('--init', action='store_true', help='初始化：获取基准数据')
    parser.add_argument('--update', action='store_true', help='周度更新：获取最新数据')
    parser.add_argument('--calculate', action='store_true', help='计算概念表现')
    parser.add_argument('--start-date', type=str, default='2024-09-24', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='结束日期')
    parser.add_argument('--min-stock-count', type=int, default=10, help='最小成分股数')
    
    args = parser.parse_args()
    
    manager = ConceptDataManager()
    
    if args.init:
        logger.info("=" * 80)
        logger.info("初始化：获取基准概念成分股数据")
        logger.info("=" * 80)
        
        # 获取概念成分股
        concept_data, failed, timeout = manager.fetch_all_concept_members()
        
        if concept_data is not None:
            # 保存数据
            manager.save_concept_data(concept_data, suffix='_base')
            
            # 打印失败和超时的概念
            if failed:
                logger.warning(f"\n获取失败的概念 ({len(failed)} 个):")
                for concept_id, concept_name in failed[:10]:
                    logger.warning(f"  - {concept_name}")
                if len(failed) > 10:
                    logger.warning(f"  ... 共 {len(failed)} 个")
            
            if timeout:
                logger.warning(f"\n获取超时的概念 ({len(timeout)} 个):")
                for concept_id, concept_name in timeout[:10]:
                    logger.warning(f"  - {concept_name}")
                if len(timeout) > 10:
                    logger.warning(f"  ... 共 {len(timeout)} 个")
    
    elif args.update:
        logger.info("=" * 80)
        logger.info("周度更新：获取最新概念成分股数据")
        logger.info("=" * 80)
        
        # 获取概念成分股
        concept_data, failed, timeout = manager.fetch_all_concept_members()
        
        if concept_data is not None:
            # 保存数据
            manager.save_concept_data(concept_data)
            
            # 打印失败和超时的概念
            if failed:
                logger.warning(f"\n获取失败的概念 ({len(failed)} 个):")
                for concept_id, concept_name in failed[:10]:
                    logger.warning(f"  - {concept_name}")
            
            if timeout:
                logger.warning(f"\n获取超时的概念 ({len(timeout)} 个):")
                for concept_id, concept_name in timeout[:10]:
                    logger.warning(f"  - {concept_name}")
            
            # 计算概念表现
            stock_data = manager.load_stock_data(args.start_date, args.end_date)
            
            if stock_data is not None:
                concept_perf, weekly_data = manager.calculate_concept_performance(
                    concept_data, stock_data, args.min_stock_count
                )
                
                # 计算评分
                concept_scored = manager.calculate_score_ranking(concept_perf, weekly_data)
                
                # 保存结果
                timestamp = datetime.now().strftime('%Y%m%d')
                perf_path = os.path.join(manager.data_dir, f'concept_performance_{timestamp}.csv')
                concept_scored.to_csv(perf_path, index=False, encoding='utf-8-sig')
                logger.info(f"概念表现已保存: {perf_path}")
                
                # 打印Top 10
                print("\n" + "=" * 100)
                print(f"综合评分 Top 10:")
                print("=" * 100)
                print(f"{'排名':<4} {'概念名称':<20} {'综合评分':<10} {'涨幅分':<8} {'回撤分':<8} {'持续分':<8} {'上榜分':<8} {'成分股分':<8}")
                print("-" * 100)
                for i, row in concept_scored.head(10).iterrows():
                    print(f"{i+1:<4} {row['concept_name']:<20} {row['total_score']:>8.1f} {row['return_score']:>7.1f} {row['drawdown_score']:>7.1f} {row['persistence_score']:>7.1f} {row['top_score']:>7.1f} {row['stock_count_score']:>7.1f}")
    
    elif args.calculate:
        logger.info("=" * 80)
        logger.info("计算概念表现")
        logger.info("=" * 80)
        
        # 加载概念数据
        concept_data = manager.load_concept_data()
        
        if concept_data is not None:
            # 加载个股数据
            stock_data = manager.load_stock_data(args.start_date, args.end_date)
            
            if stock_data is not None:
                concept_perf, weekly_data = manager.calculate_concept_performance(
                    concept_data, stock_data, args.min_stock_count
                )
                
                # 计算评分
                concept_scored = manager.calculate_score_ranking(concept_perf, weekly_data)
                
                # 保存结果
                timestamp = datetime.now().strftime('%Y%m%d')
                perf_path = os.path.join(manager.data_dir, f'concept_performance_{timestamp}.csv')
                concept_scored.to_csv(perf_path, index=False, encoding='utf-8-sig')
                logger.info(f"概念表现已保存: {perf_path}")
                
                # 打印Top 10
                print("\n" + "=" * 100)
                print(f"综合评分 Top 10:")
                print("=" * 100)
                print(f"{'排名':<4} {'概念名称':<20} {'综合评分':<10} {'涨幅分':<8} {'回撤分':<8} {'持续分':<8} {'上榜分':<8} {'成分股分':<8}")
                print("-" * 100)
                for i, row in concept_scored.head(10).iterrows():
                    print(f"{i+1:<4} {row['concept_name']:<20} {row['total_score']:>8.1f} {row['return_score']:>7.1f} {row['drawdown_score']:>7.1f} {row['persistence_score']:>7.1f} {row['top_score']:>7.1f} {row['stock_count_score']:>7.1f}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
