#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股主线逻辑识别与选股系统
基于文档chatgpt_forecast_bull_model.md实现

核心功能：
1. 板块评分系统（领涨强度、扩散速度、持续性、兑现度、拥挤度）
2. 个股优选系统（成长、质量、价格结构、流动性、估值空间）
3. 主线切换机制
4. 风控系统
5. 回测与验证
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scripts.data._shared.runtime import get_tushare_root, get_data_path

class MainlineLogicSystem:
    """主线逻辑识别系统"""
    
    def __init__(self, data_dir=None):
        """
        初始化系统
        
        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir or str(get_tushare_root())
        self.factors_dir = str(get_data_path("processed", "factors", ensure=True))
        self.sectors_dir = os.path.join(self.data_dir, 'sectors')
        self.sw_dir = os.path.join(self.data_dir, 'sw_industry')
        self.signals_dir = str(get_data_path("signals", ensure=True))
        self.concept_processed_dir = str(get_data_path("processed", "concepts", ensure=True))
        
        os.makedirs(self.factors_dir, exist_ok=True)
        os.makedirs(self.sectors_dir, exist_ok=True)
        
        # 板块评分权重（固定权重，后续可优化）
        self.sector_weights = {
            'leadership': 0.30,    # 领涨强度
            'diffusion': 0.25,     # 扩散速度
            'persistence': 0.20,   # 持续性
            'confirmation': 0.15,  # 兑现度
            'crowding': 0.10       # 拥挤度（负向）
        }
        
        # 个股评分权重
        self.stock_weights = {
            'growth': 0.25,        # 成长景气
            'quality': 0.25,       # 质量
            'price': 0.20,         # 价格结构
            'liquidity': 0.15,     # 流动性
            'valuation': 0.15      # 估值空间
        }
        
        print("主线逻辑识别系统已初始化")

        # 概念偏置权重（用于行业得分调整）
        self.concept_bias_weight = 0.15
        self.concept_overheat_penalty = 0.10
    
    def load_data(self):
        """加载数据"""
        print("\n加载数据...")
        
        # 加载日线数据
        daily_files = []
        for year in range(2020, 2027):
            filepath = os.path.join(self.data_dir, 'daily', f'daily_{year}.parquet')
            if os.path.exists(filepath):
                daily_files.append(filepath)
        
        if daily_files:
            self.daily = pd.concat([pd.read_parquet(f) for f in daily_files], ignore_index=True)
            self.daily['trade_date'] = pd.to_datetime(self.daily['trade_date'])
            print(f"  ✓ 日线数据: {len(self.daily)} 条记录")
        
        # 加载基础数据
        basic_file = os.path.join(self.data_dir, 'daily_basic_all.parquet')
        if os.path.exists(basic_file):
            self.daily_basic = pd.read_parquet(basic_file)
            self.daily_basic['trade_date'] = pd.to_datetime(self.daily_basic['trade_date'])
            print(f"  ✓ 基础数据: {len(self.daily_basic)} 条记录")
        
        # 加载财务数据
        fina_files = []
        for year in range(2020, 2026):
            filepath = os.path.join(self.data_dir, 'fundamental', f'fina_indicator_{year}.parquet')
            if os.path.exists(filepath):
                fina_files.append(filepath)
        
        if fina_files:
            self.fina_indicator = pd.concat([pd.read_parquet(f) for f in fina_files], ignore_index=True)
            print(f"  ✓ 财务数据: {len(self.fina_indicator)} 条记录")
        
        # 加载指数数据
        index_file = os.path.join(self.data_dir, 'index', 'index_ohlc_all.parquet')
        if os.path.exists(index_file):
            self.index_data = pd.read_parquet(index_file)
            self.index_data['trade_date'] = pd.to_datetime(self.index_data['date'])
            self.index_data['trade_date'] = pd.to_datetime(self.index_data['trade_date'])
            print(f"  ✓ 指数数据: {len(self.index_data)} 条记录")
        
        print("  ✓ 数据加载完成")
    
    def load_sector_data(self):
        """加载板块数据"""
        print("\n加载板块数据...")

        # 加载行业成分股（申万L1）
        industry_members_file = os.path.join(self.sw_dir, 'sw_index_member.parquet')
        industry_l1_file = os.path.join(self.sw_dir, 'sw_industry_l1.parquet')
        if os.path.exists(industry_members_file) and os.path.exists(industry_l1_file):
            members = pd.read_parquet(industry_members_file)
            l1 = pd.read_parquet(industry_l1_file)
            name_col = "industry_name" if "industry_name" in l1.columns else "index_name"
            l1 = l1.rename(columns={name_col: "industry_name"})
            members = members.rename(columns={"con_code": "ts_code"})
            self.industry_members = members.merge(
                l1[["index_code", "industry_name"]],
                on="index_code",
                how="left"
            ).dropna(subset=["industry_name", "ts_code"])
            print(f"  ✓ 行业成分股: {len(self.industry_members)} 条记录")
        else:
            print("  ! 行业成分股数据不存在，请先运行 tushare_downloader.py --task sw_industry_classify + sw_index_member")
            self.industry_members = None

        # 加载概念成分股
        concept_details_file = os.path.join(self.sectors_dir, 'concept_detail.parquet')
        if os.path.exists(concept_details_file):
            self.concept_details = pd.read_parquet(concept_details_file)
            print(f"  ✓ 概念成分股: {len(self.concept_details)} 条记录")
        else:
            print("  ! 概念成分股数据不存在，请先运行 tushare_downloader.py --task tushare_concept_detail")
            self.concept_details = None

    def build_concept_bias(self, date):
        """
        基于概念信号构建行业偏置
        返回: {sw_industry: (bias_strength, overheat_rate)}
        """
        signal_path = os.path.join(self.signals_dir, "concept_signals.parquet")
        mapping_path = os.path.join(self.concept_processed_dir, "concept_industry_primary.parquet")
        if not os.path.exists(signal_path) or not os.path.exists(mapping_path):
            return {}

        signals = pd.read_parquet(signal_path)
        if signals.empty:
            return {}
        signals["trade_date"] = pd.to_datetime(signals["trade_date"])
        signals = signals[signals["trade_date"] == pd.to_datetime(date)]
        if signals.empty:
            return {}

        signals["rank_pct"] = signals["concept_heat_score"].rank(pct=True)
        mapping = pd.read_parquet(mapping_path)
        merged = signals.merge(mapping, left_on="concept_code", right_on="concept_code", how="inner")
        if merged.empty:
            return {}

        grouped = merged.groupby("sw_industry").agg(
            mean_rank_pct=("rank_pct", "mean"),
            overheat_rate=("overheat_flag", "mean"),
        ).reset_index()

        bias_map = {}
        for _, row in grouped.iterrows():
            bias_strength = (row["mean_rank_pct"] - 0.5) * 2  # [-1,1]
            bias_map[row["sw_industry"]] = (bias_strength, row["overheat_rate"])
        return bias_map
    
    def calculate_sector_leadership(self, sector_stocks, date):
        """
        计算板块领涨强度
        
        Args:
            sector_stocks: 板块内股票列表
            date: 计算日期
            
        Returns:
            leadership_score: 领涨强度得分
        """
        # 获取板块内股票在指定日期的数据
        sector_data = self.daily[
            (self.daily['ts_code'].isin(sector_stocks)) &
            (self.daily['trade_date'] == date)
        ]
        
        if len(sector_data) == 0:
            return 0
        
        # 1. 计算板块相对收益
        sector_ret_4w = sector_data['pct_chg'].mean()
        
        # 获取指数收益
        index_data = self.index_data[self.index_data['trade_date'] == date]
        if len(index_data) > 0:
            index_ret = index_data['pct_chg'].mean()
        else:
            index_ret = 0
        
        rs = sector_ret_4w - index_ret
        
        # 2. 上涨家数占比
        up_ratio = (sector_data['pct_chg'] > 0).sum() / len(sector_data)
        
        # 3. 综合得分
        leadership_score = rs * 10 + up_ratio * 100
        
        return leadership_score
    
    def calculate_sector_diffusion(self, sector_stocks, date):
        """
        计算板块扩散速度
        
        Args:
            sector_stocks: 板块内股票列表
            date: 计算日期
            
        Returns:
            diffusion_score: 扩散速度得分
        """
        # 获取板块内股票数据
        sector_data = self.daily[
            (self.daily['ts_code'].isin(sector_stocks)) &
            (self.daily['trade_date'] <= date)
        ].sort_values(['ts_code', 'trade_date'])
        
        if len(sector_data) == 0:
            return 0
        
        # 1. 上涨占比变化率
        latest_data = sector_data.groupby('ts_code').last().reset_index()
        up_ratio = (latest_data['pct_chg'] > 0).sum() / len(latest_data)
        
        # 2. 创新高占比
        new_high_ratio = 0  # 简化版，后续可以完善
        
        # 3. 综合得分
        diffusion_score = up_ratio * 100 + new_high_ratio * 50
        
        return diffusion_score
    
    def calculate_sector_persistence(self, sector_stocks, date):
        """
        计算板块持续性
        
        Args:
            sector_stocks: 板块内股票列表
            date: 计算日期
            
        Returns:
            persistence_score: 持续性得分
        """
        # 获取板块内股票数据（最近4周）
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(weeks=4)
        
        sector_data = self.daily[
            (self.daily['ts_code'].isin(sector_stocks)) &
            (self.daily['trade_date'] >= start_date) &
            (self.daily['trade_date'] <= end_date)
        ]
        
        if len(sector_data) == 0:
            return 0
        
        # 1. 周数
        weeks = sector_data['trade_date'].dt.to_period('W').nunique()
        
        # 2. 相对强度为正的时间占比（简化版）
        # 后续完善
        
        # 3. 综合得分
        persistence_score = weeks * 25  # 每周25分
        
        return persistence_score
    
    def calculate_sector_confirmation(self, sector_stocks, date):
        """
        计算板块兑现度
        
        Args:
            sector_stocks: 板块内股票列表
            date: 计算日期
            
        Returns:
            confirmation_score: 兑现度得分
        """
        # 获取板块内股票的财务数据
        sector_fina = self.fina_indicator[
            (self.fina_indicator['ts_code'].isin(sector_stocks)) &
            (self.fina_indicator['end_date'] <= date)
        ]
        
        if len(sector_fina) == 0:
            return 0
        
        # 1. 利润同比改善比例
        latest_fina = sector_fina.groupby('ts_code').last().reset_index()
        profit_improve_ratio = (latest_fina['np_yoy'] > 0).sum() / len(latest_fina)
        
        # 2. ROE抬升比例
        roe_improve_ratio = 0  # 简化版
        
        # 3. 综合得分
        confirmation_score = profit_improve_ratio * 100
        
        return confirmation_score
    
    def calculate_sector_crowding(self, sector_stocks, date):
        """
        计算板块拥挤度
        
        Args:
            sector_stocks: 板块内股票列表
            date: 计算日期
            
        Returns:
            crowding_score: 拥挤度得分（越高越拥挤）
        """
        # 获取板块内股票的基础数据
        sector_basic = self.daily_basic[
            (self.daily_basic['ts_code'].isin(sector_stocks)) &
            (self.daily_basic['trade_date'] == date)
        ]
        
        if len(sector_basic) == 0:
            return 0
        
        # 1. 换手率异常
        avg_turnover = sector_basic['turnover_rate'].mean()
        
        # 2. 综合得分
        crowding_score = avg_turnover * 10
        
        return crowding_score
    
    def calculate_sector_score(self, sector_stocks, date, sector_name, concept_bias=None):
        """
        计算板块综合评分
        
        Args:
            sector_stocks: 板块内股票列表
            date: 计算日期
            sector_name: 板块名称
            
        Returns:
            sector_score: 板块综合评分
            details: 详细得分
        """
        # 计算五大指标
        leadership = self.calculate_sector_leadership(sector_stocks, date)
        diffusion = self.calculate_sector_diffusion(sector_stocks, date)
        persistence = self.calculate_sector_persistence(sector_stocks, date)
        confirmation = self.calculate_sector_confirmation(sector_stocks, date)
        crowding = self.calculate_sector_crowding(sector_stocks, date)
        
        # 综合评分
        sector_score = (
            leadership * self.sector_weights['leadership'] +
            diffusion * self.sector_weights['diffusion'] +
            persistence * self.sector_weights['persistence'] +
            confirmation * self.sector_weights['confirmation'] -
            crowding * self.sector_weights['crowding']
        )

        bias_strength = 0
        overheat_rate = 0
        if concept_bias:
            bias_strength, overheat_rate = concept_bias
            sector_score = sector_score * (1 + self.concept_bias_weight * bias_strength)
            if overheat_rate >= 0.3:
                sector_score = sector_score * (1 - self.concept_overheat_penalty)
        
        details = {
            'sector_name': sector_name,
            'leadership': leadership,
            'diffusion': diffusion,
            'persistence': persistence,
            'confirmation': confirmation,
            'crowding': crowding,
            'concept_bias': bias_strength,
            'concept_overheat_rate': overheat_rate,
            'total_score': sector_score
        }
        
        return sector_score, details
    
    def rank_sectors(self, date, top_k=3):
        """
        对所有板块进行排序
        
        Args:
            date: 计算日期
            top_k: 选择Top K个板块
            
        Returns:
            top_sectors: Top K板块
        """
        print(f"\n计算板块评分 ({date})...")
        
        sector_scores = []

        concept_bias_map = self.build_concept_bias(date)
        
        # 如果有行业成分股数据
        if self.industry_members is not None:
            # 按行业分组
            for industry_name, group in self.industry_members.groupby('industry_name'):
                stocks = group['con_code'].tolist()
                
                bias = concept_bias_map.get(industry_name)
                score, details = self.calculate_sector_score(stocks, date, industry_name, concept_bias=bias)
                sector_scores.append(details)
                
                print(f"  {industry_name}: {score:.2f}")
        
        # 如果有概念成分股数据
        if self.concept_details is not None:
            # 按概念分组
            for concept_name, group in self.concept_details.groupby('concept_name'):
                stocks = group['ts_code'].tolist()
                
                score, details = self.calculate_sector_score(stocks, date, concept_name)
                sector_scores.append(details)
        
        if not sector_scores:
            print("  ! 没有板块数据")
            return None
        
        # 排序
        sector_df = pd.DataFrame(sector_scores)
        sector_df = sector_df.sort_values('total_score', ascending=False)
        
        # 选择Top K
        top_sectors = sector_df.head(top_k)
        
        print(f"\n  ✓ Top {top_k}板块:")
        for idx, row in top_sectors.iterrows():
            print(f"    {idx+1}. {row['sector_name']}: {row['total_score']:.2f}")
        
        return top_sectors
    
    def calculate_stock_score(self, ts_code, date):
        """
        计算个股综合评分
        
        Args:
            ts_code: 股票代码
            date: 计算日期
            
        Returns:
            stock_score: 个股综合评分
        """
        # 获取个股数据
        stock_data = self.daily[
            (self.daily['ts_code'] == ts_code) &
            (self.daily['trade_date'] <= date)
        ].sort_values('trade_date')
        
        if len(stock_data) < 20:
            return 0
        
        # 获取基础数据
        stock_basic = self.daily_basic[
            (self.daily_basic['ts_code'] == ts_code) &
            (self.daily_basic['trade_date'] == date)
        ]
        
        if len(stock_basic) == 0:
            return 0
        
        # 1. 成长景气（简化版）
        growth_score = stock_data['pct_chg'].tail(20).mean() * 10
        
        # 2. 质量（简化版）
        quality_score = 50  # 后续完善
        
        # 3. 价格结构
        price_score = stock_data['pct_chg'].tail(5).mean() * 10
        
        # 4. 流动性
        liquidity_score = min(stock_basic['amount'].values[0] / 100000000, 100)  # 成交额(亿)
        
        # 5. 估值空间
        pe = stock_basic['pe_ttm'].values[0]
        if pe > 0:
            valuation_score = 100 / pe  # PE越低，得分越高
        else:
            valuation_score = 0
        
        # 综合评分
        stock_score = (
            growth_score * self.stock_weights['growth'] +
            quality_score * self.stock_weights['quality'] +
            price_score * self.stock_weights['price'] +
            liquidity_score * self.stock_weights['liquidity'] +
            valuation_score * self.stock_weights['valuation']
        )
        
        return stock_score
    
    def select_stocks_in_sectors(self, top_sectors, date, stocks_per_sector=5):
        """
        在Top板块内选股
        
        Args:
            top_sectors: Top板块
            date: 计算日期
            stocks_per_sector: 每个板块选多少只股票
            
        Returns:
            selected_stocks: 选出的股票
        """
        print(f"\n在Top板块内选股...")
        
        selected_stocks = []
        
        for idx, sector_row in top_sectors.iterrows():
            sector_name = sector_row['sector_name']
            
            # 获取板块内股票
            if self.industry_members is not None and sector_name in self.industry_members['industry_name'].values:
                stocks = self.industry_members[self.industry_members['industry_name'] == sector_name]['con_code'].tolist()
            elif self.concept_details is not None and sector_name in self.concept_details['concept_name'].values:
                stocks = self.concept_details[self.concept_details['concept_name'] == sector_name]['ts_code'].tolist()
            else:
                continue
            
            # 计算每只股票的评分
            stock_scores = []
            for ts_code in stocks:
                score = self.calculate_stock_score(ts_code, date)
                stock_scores.append({
                    'ts_code': ts_code,
                    'score': score,
                    'sector': sector_name
                })
            
            # 排序并选择Top N
            stock_df = pd.DataFrame(stock_scores)
            stock_df = stock_df.sort_values('score', ascending=False)
            top_stocks = stock_df.head(stocks_per_sector)
            
            selected_stocks.extend(top_stocks.to_dict('records'))
            
            print(f"  {sector_name}: 选出{len(top_stocks)}只股票")
        
        result_df = pd.DataFrame(selected_stocks)
        print(f"\n  ✓ 共选出{len(result_df)}只股票")
        
        return result_df
    
    def run(self, start_date='2024-01-01', end_date='2024-12-31', frequency='W'):
        """
        运行主线逻辑识别系统
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            frequency: 频率（W=周频）
        """
        print("=" * 70)
        print("主线逻辑识别系统启动")
        print("=" * 70)
        
        # 加载数据
        self.load_data()
        self.load_sector_data()
        
        # 生成日期序列
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        all_results = []
        
        for date in date_range:
            print(f"\n{'='*70}")
            print(f"处理日期: {date.strftime('%Y-%m-%d')}")
            print(f"{'='*70}")
            
            # 1. 板块评分
            top_sectors = self.rank_sectors(date, top_k=3)
            
            if top_sectors is None:
                continue
            
            # 2. 板块内选股
            selected_stocks = self.select_stocks_in_sectors(top_sectors, date, stocks_per_sector=5)
            
            # 3. 保存结果
            for idx, stock_row in selected_stocks.iterrows():
                all_results.append({
                    'trade_date': date,
                    'ts_code': stock_row['ts_code'],
                    'sector': stock_row['sector'],
                    'stock_score': stock_row['score']
                })
        
        # 保存结果
        if all_results:
            results_df = pd.DataFrame(all_results)
            output_file = os.path.join(self.factors_dir, 'mainline_selection_results.csv')
            results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n{'='*70}")
            print(f"✓ 结果已保存到: {output_file}")
            print(f"{'='*70}")
            
            return results_df
        else:
            print("\n! 没有生成任何结果")
            return None

if __name__ == '__main__':
    # 创建系统实例
    system = MainlineLogicSystem()
    
    # 运行系统
    results = system.run(
        start_date='2024-01-01',
        end_date='2024-12-31',
        frequency='W'
    )
    
    if results is not None:
        print(f"\n总结果: {len(results)} 条记录")
        print(results.head())
