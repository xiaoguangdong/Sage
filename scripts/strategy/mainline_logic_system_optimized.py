#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股主线逻辑识别与选股系统（优化版）
基于文档chatgpt_forecast_bull_model.md实现

核心优化：
1. 完善五大板块指标（领涨强度、扩散速度、持续性、兑现度、拥挤度）
2. 完善个股五维评分（成长、质量、价格结构、流动性、估值空间）
3. 使用向量化操作提高性能
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scripts.data._shared.runtime import get_tushare_root, get_data_path, get_project_root

class MainlineLogicSystemOptimized:
    """主线逻辑识别系统（优化版）"""
    
    def __init__(self, data_dir=None):
        """
        初始化系统
        
        Args:
            data_dir: 数据目录
        """
        primary_root = Path(data_dir) if data_dir else get_tushare_root()
        legacy_root = get_project_root() / "data" / "tushare"
        self.data_roots = [str(primary_root)]
        if str(legacy_root) != str(primary_root):
            self.data_roots.append(str(legacy_root))

        self.data_dir = self.data_roots[0]
        self.factors_dir = str(get_data_path("processed", "factors", ensure=True))
        self.sectors_dir = self._resolve_dir("sectors")
        
        os.makedirs(self.factors_dir, exist_ok=True)
        os.makedirs(self.sectors_dir, exist_ok=True)
        
        # 板块评分权重（优化版）
        self.sector_weights = {
            'leadership': 0.30,    # 领涨强度
            'diffusion': 0.25,     # 扩散速度
            'persistence': 0.20,   # 持续性
            'confirmation': 0.15,  # 兑现度
            'crowding': 0.10       # 拥挤度（负向）
        }
        
        # 个股评分权重（优化版）
        self.stock_weights = {
            'growth': 0.25,        # 成长景气
            'quality': 0.25,       # 质量
            'price': 0.20,         # 价格结构
            'liquidity': 0.15,     # 流动性
            'valuation': 0.15      # 估值空间
        }
        
        print("主线逻辑识别系统（优化版）已初始化")

    def _resolve_dir(self, subdir):
        for root in self.data_roots:
            candidate = os.path.join(root, subdir)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(self.data_roots[0], subdir)

    def _resolve_file(self, *parts):
        for root in self.data_roots:
            candidate = os.path.join(root, *parts)
            if os.path.exists(candidate):
                return candidate
        return os.path.join(self.data_roots[0], *parts)

    @staticmethod
    def _looks_like_concept_code(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(50)
        if sample.empty:
            return False
        return sample.str.endswith(".TI").mean() > 0.6

    def _normalize_ths_member(self, members: pd.DataFrame, index_df: pd.DataFrame | None = None) -> pd.DataFrame:
        cols = members.columns.tolist()
        concept_code_col = None
        stock_col = None

        for candidate in ["concept_code", "index_code", "id", "ts_code", "code"]:
            if candidate in cols and self._looks_like_concept_code(members[candidate]):
                concept_code_col = candidate
                break
        if concept_code_col is None:
            for candidate in ["concept_code", "index_code", "id", "ts_code", "code"]:
                if candidate in cols:
                    concept_code_col = candidate
                    break

        for candidate in ["ts_code", "con_code", "stock_code", "code", "symbol"]:
            if candidate in cols and candidate != concept_code_col:
                stock_col = candidate
                break

        if not concept_code_col or not stock_col:
            raise ValueError(f"ths_member 字段不足，无法识别概念和成分列: {cols}")

        concept_name_col = "concept_name" if "concept_name" in cols else None
        if not concept_name_col:
            for candidate in ["index_name", "name"]:
                if candidate in cols:
                    concept_name_col = candidate
                    break

        normalized = members.rename(columns={concept_code_col: "concept_code", stock_col: "ts_code"}).copy()
        normalized["concept_code"] = normalized["concept_code"].astype(str)
        normalized["ts_code"] = normalized["ts_code"].astype(str)

        if concept_name_col:
            normalized = normalized.rename(columns={concept_name_col: "concept_name"})

        if "concept_name" not in normalized.columns and index_df is not None and not index_df.empty:
            name_col = "name" if "name" in index_df.columns else ("index_name" if "index_name" in index_df.columns else None)
            if name_col and "ts_code" in index_df.columns:
                names = index_df[["ts_code", name_col]].rename(columns={"ts_code": "concept_code", name_col: "concept_name"})
                normalized = normalized.merge(names, on="concept_code", how="left")

        if "concept_name" not in normalized.columns:
            normalized["concept_name"] = normalized["concept_code"]

        return normalized[["concept_code", "concept_name", "ts_code"]].drop_duplicates()
    
    def load_data(self, start_date, end_date):
        """
        加载数据（优化版，只加载需要的年份）
        """
        print("\n加载数据...")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 加载日线数据
        daily_files = []
        for year in range(start_dt.year, end_dt.year + 1):
            filepath = self._resolve_file('daily', f'daily_{year}.parquet')
            if os.path.exists(filepath):
                daily_files.append(filepath)
        
        if daily_files:
            self.daily = pd.concat([pd.read_parquet(f) for f in daily_files], ignore_index=True)
            self.daily['trade_date'] = pd.to_datetime(self.daily['trade_date'])
            print(f"  ✓ 日线数据: {len(self.daily)} 条记录")
        
        # 加载基础数据
        basic_file = self._resolve_file('daily_basic_all.parquet')
        if os.path.exists(basic_file):
            self.daily_basic = pd.read_parquet(basic_file)
            self.daily_basic['trade_date'] = pd.to_datetime(self.daily_basic['trade_date'])
            print(f"  ✓ 基础数据: {len(self.daily_basic)} 条记录")
        
        # 加载财务数据
        fina_files = []
        for year in range(start_dt.year, end_dt.year + 1):
            filepath = self._resolve_file('fundamental', f'fina_indicator_{year}.parquet')
            if os.path.exists(filepath):
                fina_files.append(filepath)
        
        if fina_files:
            self.fina_indicator = pd.concat([pd.read_parquet(f) for f in fina_files], ignore_index=True)
            self.fina_indicator['end_date'] = pd.to_datetime(self.fina_indicator['end_date'])
            print(f"  ✓ 财务数据: {len(self.fina_indicator)} 条记录")
        
        # 加载指数数据
        index_file = self._resolve_file('index', 'index_ohlc_all.parquet')
        if os.path.exists(index_file):
            self.index_data = pd.read_parquet(index_file)
            self.index_data['trade_date'] = pd.to_datetime(self.index_data['date'])
            print(f"  ✓ 指数数据: {len(self.index_data)} 条记录")
        
        print("  ✓ 数据加载完成")
    
    def load_sector_data(self):
        """加载板块数据"""
        print("\n加载板块数据...")
        
        # 加载行业成分股
        industry_members_file = os.path.join(self.sectors_dir, 'all_index_members.csv')
        if os.path.exists(industry_members_file):
            self.industry_members = pd.read_csv(industry_members_file)
            print(f"  ✓ 行业成分股: {len(self.industry_members)} 条记录")
        else:
            print("  ! 行业成分股数据不存在")
            self.industry_members = None
        
        # 加载概念成分股（ths_member）
        ths_member_file = os.path.join(self._resolve_dir('concepts'), 'ths_member.parquet')
        ths_index_file = os.path.join(self._resolve_dir('concepts'), 'ths_index.parquet')
        if os.path.exists(ths_member_file):
            member_df = pd.read_parquet(ths_member_file)
            index_df = pd.read_parquet(ths_index_file) if os.path.exists(ths_index_file) else None
            self.concept_details = self._normalize_ths_member(member_df, index_df=index_df)
            print(f"  ✓ 同花顺概念成分股: {len(self.concept_details)} 条记录")
        else:
            print("  ! 同花顺概念成分数据不存在")
            self.concept_details = None
        
        print(f"  ✓ 板块数据加载完成：")
        if self.industry_members is not None:
            print(f"    - 行业板块: {self.industry_members['industry_name'].nunique()} 个")
        if self.concept_details is not None:
            print(f"    - 概念板块: {self.concept_details['concept_name'].nunique()} 个")
    
    def calculate_sector_leadership(self, sector_stocks, date):
        """
        计算板块领涨强度（优化版）
        
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
        
        # 1. 相对强度 RS_g = Return_{g,4w} - Return_{Index,4w}
        sector_ret = sector_data['pct_chg'].mean()
        
        # 获取指数收益
        index_data = self.index_data[
            (self.index_data['trade_date'] == date) & 
            (self.index_data['index_name'].isin(['沪深300', '中证500']))
        ]
        if len(index_data) > 0:
            index_ret = index_data['pct_chg'].mean()
        else:
            index_ret = 0
        
        rs = sector_ret - index_ret
        
        # 2. 上行一致性：上涨占比
        up_ratio = (sector_data['pct_chg'] > 0).mean()
        
        # 3. 涨停强度（简化版：涨跌幅>9.5%的比例）
        limit_up_ratio = (sector_data['pct_chg'] > 9.5).mean()
        
        # 综合得分
        leadership_score = rs * 20 + up_ratio * 100 + limit_up_ratio * 50
        
        return leadership_score
    
    def calculate_sector_diffusion(self, sector_stocks, date):
        """
        计算板块扩散速度（优化版）
        
        Args:
            sector_stocks: 板块内股票列表
            date: 计算日期
            
        Returns:
            diffusion_score: 扩散速度得分
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
        
        # 1. 上涨占比变化率
        latest_data = sector_data[sector_data['trade_date'] == date]
        if len(latest_data) > 0:
            up_ratio = (latest_data['pct_chg'] > 0).mean()
        else:
            up_ratio = 0
        
        # 2. 创新高占比（简化版：价格接近近期高点）
        if len(latest_data) > 0 and 'close' in latest_data.columns:
            # 计算过去60天的高点
            historical_data = sector_data[sector_data['ts_code'].isin(latest_data['ts_code'])]
            if len(historical_data) > 0:
                max_prices = historical_data.groupby('ts_code')['close'].max().reset_index()
                merged = latest_data.merge(max_prices, on='ts_code', suffixes=('', '_max'))
                new_high_ratio = (merged['close'] >= merged['close_max'] * 0.98).mean()
            else:
                new_high_ratio = 0
        else:
            new_high_ratio = 0
        
        # 3. 市值梯度扩散（简化版）
        # 后续可以实现从大盘股到小盘股的扩散
        
        # 综合得分
        diffusion_score = up_ratio * 100 + new_high_ratio * 50
        
        return diffusion_score
    
    def calculate_sector_persistence(self, sector_stocks, date):
        """
        计算板块持续性（优化版）
        
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
        
        # 1. 每周上涨比例
        weekly_data = sector_data.groupby([pd.Grouper(key='trade_date', freq='W'), 'ts_code']).last()
        up_ratios = weekly_data.groupby('trade_date').apply(lambda x: (x['pct_chg'] > 0).mean())
        
        # 2. 相对强度为正的时间占比
        positive_weeks = (up_ratios > 0.5).sum() / len(up_ratios) if len(up_ratios) > 0 else 0
        
        # 3. 综合得分
        persistence_score = positive_weeks * 100
        
        return persistence_score
    
    def calculate_sector_confirmation(self, sector_stocks, date):
        """
        计算板块兑现度（优化版）
        
        Args:
            sector_stocks: 板块内股票列表
            date: 计算日期
            
        Returns:
            confirmation_score: 兑现度得分
        """
        # 获取板块内股票的财务数据
        sector_fina = self.fina_indicator[
            (self.fina_indicator['ts_code'].isin(sector_stocks)) &
            (self.fina_indicator['end_date'] <= pd.to_datetime(date))
        ]
        
        if len(sector_fina) == 0:
            return 0
        
        # 1. 利润同比改善比例
        latest_fina = sector_fina.groupby('ts_code').last().reset_index()
        
        if 'netprofit_yoy' in latest_fina.columns:
            profit_improve_ratio = (latest_fina['netprofit_yoy'] > 0).mean()
        else:
            profit_improve_ratio = 0
        
        # 2. ROE抬升比例
        if 'roe_yoy' in latest_fina.columns:
            roe_improve_ratio = (latest_fina['roe_yoy'] > 0).mean()
        else:
            roe_improve_ratio = 0
        
        # 3. 现金流改善比例
        if 'ocf_yoy' in latest_fina.columns:
            ocf_improve_ratio = (latest_fina['ocf_yoy'] > 0).mean()
        else:
            ocf_improve_ratio = 0
        
        # 综合得分
        confirmation_score = (
            profit_improve_ratio * 100 * 0.5 +
            roe_improve_ratio * 100 * 0.3 +
            ocf_improve_ratio * 100 * 0.2
        )
        
        return confirmation_score
    
    def calculate_sector_crowding(self, sector_stocks, date):
        """
        计算板块拥挤度（优化版）
        
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
        
        # 1. 换手率异常（Z-score）
        avg_turnover = sector_basic['turnover_rate'].mean()
        turnover_std = sector_basic['turnover_rate'].std()
        turnover_zscore = (avg_turnover / turnover_std) if turnover_std > 0 else 0
        
        # 2. 成交额占比（简化版）
        total_amount = sector_basic['total_mv'].sum()
        
        # 3. 收益相关性（简化版，后续可以计算板块内股票收益的相关性）
        
        # 综合得分
        crowding_score = avg_turnover * 10 + turnover_zscore * 5
        
        return crowding_score
    
    def calculate_sector_score(self, sector_stocks, date, sector_name):
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
        
        details = {
            'sector_name': sector_name,
            'leadership': leadership,
            'diffusion': diffusion,
            'persistence': persistence,
            'confirmation': confirmation,
            'crowding': crowding,
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
        
        # 处理行业板块
        if self.industry_members is not None:
            for industry_name, group in self.industry_members.groupby('industry_name'):
                stocks = group['con_code'].tolist()
                
                score, details = self.calculate_sector_score(stocks, date, industry_name)
                sector_scores.append(details)
        
        # 处理概念板块
        if self.concept_details is not None:
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
            print(f"    {idx+1}. {row['sector_name']}: {row['total_score']:.2f} "
                  f"(L:{row['leadership']:.2f} D:{row['diffusion']:.2f} "
                  f"P:{row['persistence']:.2f} C:{row['confirmation']:.2f} Cr:{row['crowding']:.2f})")
        
        return top_sectors
    
    def calculate_stock_growth_score(self, ts_code, date):
        """
        计算个股成长景气得分（优化版）
        """
        # 获取个股数据
        stock_data = self.daily[
            (self.daily['ts_code'] == ts_code) &
            (self.daily['trade_date'] <= date)
        ].tail(60)  # 最近60天
        
        if len(stock_data) < 20:
            return 0
        
        # 1. 4周收益
        if len(stock_data) >= 20:
            ret_4w = (stock_data.iloc[-1]['close'] / stock_data.iloc[-20]['close'] - 1) * 100
        else:
            ret_4w = 0
        
        # 2. 12周收益
        if len(stock_data) >= 60:
            ret_12w = (stock_data.iloc[-1]['close'] / stock_data.iloc[-60]['close'] - 1) * 100
        else:
            ret_12w = 0
        
        # 3. 收益加速度（近4周收益 - 前4周收益）
        if len(stock_data) >= 40:
            ret_acc = ret_4w - (stock_data.iloc[-20]['close'] / stock_data.iloc[-40]['close'] - 1) * 100
        else:
            ret_acc = 0
        
        # 综合得分
        growth_score = (
            ret_4w * 0.4 +
            ret_12w * 0.4 +
            ret_acc * 0.2
        )
        
        return growth_score
    
    def calculate_stock_quality_score(self, ts_code, date):
        """
        计算个股质量得分（优化版）
        """
        # 获取财务数据
        stock_fina = self.fina_indicator[
            (self.fina_indicator['ts_code'] == ts_code) &
            (self.fina_indicator['end_date'] <= pd.to_datetime(date))
        ]
        
        if len(stock_fina) == 0:
            return 50  # 默认值
        
        latest_fina = stock_fina.groupby('ts_code').last().reset_index().iloc[0]
        
        # 1. ROE
        roe = latest_fina.get('roe', 0)
        if pd.isna(roe):
            roe = 0
        
        # 2. 毛利率
        gross_margin = latest_fina.get('gross_margin', 0)
        if pd.isna(gross_margin):
            gross_margin = 0
        
        # 3. 资产负债率（负债率越低越好）
        debt_to_assets = latest_fina.get('debt_to_assets', 0)
        if pd.isna(debt_to_assets):
            debt_to_assets = 0
        
        # 综合得分
        quality_score = (
            (roe * 2) +
            (gross_margin * 2) +
            ((1 - debt_to_assets) * 100)
        )
        
        return quality_score
    
    def calculate_stock_price_score(self, ts_code, date):
        """
        计算个股价格结构得分（优化版）
        """
        # 获取个股数据
        stock_data = self.daily[
            (self.daily['ts_code'] == ts_code) &
            (self.daily['trade_date'] <= date)
        ].tail(60)
        
        if len(stock_data) < 20:
            return 0
        
        # 1. 是否突破120日新高
        if len(stock_data) >= 120:
            max_120d = stock_data['close'].tail(120).max()
            breakout = 1 if stock_data.iloc[-1]['close'] >= max_120d * 0.98 else 0
        else:
            breakout = 0
        
        # 2. 回撤控制
        max_price = stock_data['close'].tail(20).max()
        current_price = stock_data.iloc[-1]['close']
        drawdown = (max_price - current_price) / max_price if max_price > 0 else 0
        drawdown_score = max(0, 100 - drawdown * 1000)  # 回撤越小得分越高
        
        # 3. 收益/波动比
        if len(stock_data) >= 20:
            ret = (stock_data.iloc[-1]['close'] / stock_data.iloc[-20]['close'] - 1) * 100
            vol = stock_data['pct_chg'].tail(20).std()
            sharpe = ret / vol if vol > 0 else 0
        else:
            sharpe = 0
        
        # 综合得分
        price_score = (
            breakout * 30 +
            drawdown_score * 0.3 +
            sharpe * 10
        )
        
        return price_score
    
    def calculate_stock_liquidity_score(self, ts_code, date):
        """
        计算个股流动性得分（优化版）
        """
        # 获取基础数据
        stock_basic = self.daily_basic[
            (self.daily_basic['ts_code'] == ts_code) &
            (self.daily_basic['trade_date'] == date)
        ]
        
        if len(stock_basic) == 0:
            return 0
        
        basic = stock_basic.iloc[0]
        
        # 1. 换手率（适中为好）
        turnover = basic.get('turnover_rate', 0)
        if pd.isna(turnover):
            turnover = 0
        
        # 换手率在2-10%之间得分为100
        if 2 <= turnover <= 10:
            liquidity = 100
        elif turnover < 2:
            liquidity = turnover * 50
        else:
            liquidity = max(0, 100 - (turnover - 10) * 10)
        
        # 2. 市值（流动性好）
        total_mv = basic.get('total_mv', 0)
        if pd.isna(total_mv):
            total_mv = 0
        
        mv_score = min(100, total_mv / 100000000)  # 转换为亿，上限100
        
        # 综合得分
        liquidity_score = liquidity * 0.7 + mv_score * 0.3
        
        return liquidity_score
    
    def calculate_stock_valuation_score(self, ts_code, date):
        """
        计算个股估值空间得分（优化版）
        """
        # 获取基础数据
        stock_basic = self.daily_basic[
            (self.daily_basic['ts_code'] == ts_code) &
            (self.daily_basic['trade_date'] == date)
        ]
        
        if len(stock_basic) == 0:
            return 0
        
        basic = stock_basic.iloc[0]
        
        # 1. PE估值
        pe = basic.get('pe_ttm', 0)
        if pd.isna(pe) or pe <= 0:
            pe_score = 0
        else:
            # PE越低得分越高，PE在10-30之间得分为100
            if 10 <= pe <= 30:
                pe_score = 100
            elif pe < 10:
                pe_score = pe * 10
            else:
                pe_score = max(0, 100 - (pe - 30) * 2)
        
        # 2. PB估值
        pb = basic.get('pb', 0)
        if pd.isna(pb) or pb <= 0:
            pb_score = 0
        else:
            # PB在1-3之间得分为100
            if 1 <= pb <= 3:
                pb_score = 100
            elif pb < 1:
                pb_score = pb * 100
            else:
                pb_score = max(0, 100 - (pb - 3) * 20)
        
        # 综合得分
        valuation_score = pe_score * 0.6 + pb_score * 0.4
        
        return valuation_score
    
    def calculate_stock_score(self, ts_code, date):
        """
        计算个股综合评分（优化版）
        
        Args:
            ts_code: 股票代码
            date: 计算日期
            
        Returns:
            stock_score: 个股综合评分
        """
        # 计算五个维度
        growth_score = self.calculate_stock_growth_score(ts_code, date)
        quality_score = self.calculate_stock_quality_score(ts_code, date)
        price_score = self.calculate_stock_price_score(ts_code, date)
        liquidity_score = self.calculate_stock_liquidity_score(ts_code, date)
        valuation_score = self.calculate_stock_valuation_score(ts_code, date)
        
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
            stocks = None
            
            # 从行业成分股获取
            if self.industry_members is not None and sector_name in self.industry_members['industry_name'].values:
                stocks = self.industry_members[self.industry_members['industry_name'] == sector_name]['con_code'].tolist()
            # 从概念成分股获取
            elif self.concept_details is not None and sector_name in self.concept_details['concept_name'].values:
                stocks = self.concept_details[self.concept_details['concept_name'] == sector_name]['ts_code'].tolist()
            
            if stocks is None or len(stocks) == 0:
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
        print("主线逻辑识别系统（优化版）启动")
        print("=" * 70)
        
        # 加载数据
        self.load_data(start_date, end_date)
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
            output_file = os.path.join(self.factors_dir, 'mainline_selection_results_optimized.csv')
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
    system = MainlineLogicSystemOptimized()
    
    # 运行系统（测试一个日期）
    results = system.run(
        start_date='2024-06-01',
        end_date='2024-06-01',
        frequency='D'
    )
    
    if results is not None:
        print(f"\n总结果: {len(results)} 条记录")
        print(results.head())
