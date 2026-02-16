"""
价格特征（动量、波动、量价）

按照豆包 Champion 策略四因子体系：
- 质量因子（30%）：在 fundamental_features 计算
- 成长因子（30%）：在 fundamental_features 计算
- 动量因子（20%）：相对板块超额收益、成交额分位数
- 低波动因子（20%）：波动率、最大回撤、下行波动率
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging

from .base import FeatureGenerator, FeatureSpec
from .registry import register_feature

logger = logging.getLogger(__name__)

# 豆包 Champion 策略四因子权重
DOUBAO_FACTOR_WEIGHTS = {
    'quality': 0.30,    # 质量因子（在 fundamental_features）
    'growth': 0.30,     # 成长因子（在 fundamental_features）
    'momentum': 0.20,   # 动量因子
    'low_vol': 0.20,    # 低波动因子
}


@register_feature
class PriceFeatures(FeatureGenerator):
    """价格特征提取器（豆包 Champion 策略四因子体系）"""

    spec = FeatureSpec(
        name="price_features",
        input_fields=("date", "stock", "close"),
        description="个股价格相关特征（动量/低波动）- 豆包四因子体系",
    )
    
    def __init__(
        self,
        momentum_periods: List[int] = None,
        vol_periods: List[int] = None,
    ):
        """
        初始化价格特征提取器

        Args:
            momentum_periods: 动量计算周期（默认 [20, 60] 日）
            vol_periods: 波动率计算周期（默认 [20, 60] 日）
        """
        self.momentum_periods = momentum_periods or [20, 60]
        self.vol_periods = vol_periods or [20, 60]
    
    def calculate_momentum_features(
        self,
        df: pd.DataFrame,
        periods: List[int] = None,
        industry_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算动量特征（豆包动量因子，权重20%）

        动量因子包括：
        - 过去N日动量
        - 相对行业超额收益（个股动量 - 行业动量）
        - 成交额分位数（资金关注度）

        Args:
            df: 包含收盘价的DataFrame
            periods: 周期列表
            industry_data: 行业指数数据（用于计算相对超额收益）

        Returns:
            包含动量特征的DataFrame
        """
        periods = periods or self.momentum_periods
        df = df.copy()
        df = df.sort_values(['stock', 'date'])
        close_group = df.groupby('stock')['close']

        for period in periods:
            # 1. 价格动量
            df[f'momentum_{period}d'] = close_group.pct_change(period)

        # 2. 相对行业超额收益（豆包动量因子核心）
        # 个股动量 - 行业动量
        if industry_data is not None and not industry_data.empty:
            # 合并行业数据
            if 'industry_code' in df.columns:
                df = df.merge(
                    industry_data[['date', 'industry_code', 'industry_ret_20d', 'industry_ret_60d']],
                    on=['date', 'industry_code'],
                    how='left'
                )
                # 计算相对超额收益
                for period in periods:
                    mom_col = f'momentum_{period}d'
                    ind_col = f'industry_ret_{period}d'
                    if mom_col in df.columns and ind_col in df.columns:
                        df[f'excess_return_vs_industry_{period}d'] = df[mom_col] - df[ind_col]
        else:
            # 如果没有行业数据，使用默认值
            for period in periods:
                df[f'excess_return_vs_industry_{period}d'] = df.get(f'momentum_{period}d', 0)

        # 3. 成交额分位数（豆包动量因子 - 资金关注度）
        if 'amount' in df.columns:
            # 个股成交额在全市场的分位数
            df['amt_quantile'] = df.groupby('date')['amount'].transform(
                lambda x: x.rank(pct=True)
            )
            # 过去20日成交额均值 / 过去250日成交额均值（放倍量判断）
            df['amt_ratio_20_250'] = df.groupby('stock')['amount'].transform(
                lambda s: s.rolling(20).mean() / (s.rolling(250).mean() + 1e-8)
            )

        # 动量因子综合评分
        mom_components = []
        
        # 相对行业超额收益 z-score
        for period in periods:
            col = f'excess_return_vs_industry_{period}d'
            if col in df.columns:
                df[f'{col}_zscore'] = df.groupby('stock')[col].transform(
                    lambda x: (x - x.rolling(252, min_periods=60).mean()) / 
                             (x.rolling(252, min_periods=60).std() + 1e-8)
                )
                mom_components.append(f'{col}_zscore')

        # 成交额分位数 z-score
        if 'amt_quantile' in df.columns:
            df['amt_quantile_zscore'] = df.groupby('stock')['amt_quantile'].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / 
                         (x.rolling(252, min_periods=60).std() + 1e-8)
            )
            mom_components.append('amt_quantile_zscore')

        # 动量因子综合评分
        if mom_components:
            # 豆包方案：相对超额收益权重更高
            momentum_weights = {}
            for col in mom_components:
                if 'excess_return' in col:
                    momentum_weights[col] = 0.6 / max(1, len([c for c in mom_components if 'excess_return' in c]))
                else:
                    momentum_weights[col] = 0.4 / max(1, len([c for c in mom_components if 'excess_return' not in c]))
            
            df['momentum_score'] = sum(
                df.get(c, 0) * momentum_weights.get(c, 1/len(mom_components))
                for c in mom_components
            )
        else:
            df['momentum_score'] = np.nan

        logger.info(f"动量特征计算完成，特征数: {len(periods) + len(mom_components) + 2}")

        return df
    
    def calculate_liquidity_features(
        self,
        df: pd.DataFrame,
        periods: List[int] = [4, 12]
    ) -> pd.DataFrame:
        """
        计算流动性特征
        
        Args:
            df: 包含成交量和市值的DataFrame
            periods: 周期列表
            
        Returns:
            包含流动性特征的DataFrame
        """
        df = df.copy()
        df = df.sort_values(['stock', 'date'])
        close_group = df.groupby('stock')['close']
        volume_group = df.groupby('stock')['volume'] if 'volume' in df.columns else None
        amount_group = df.groupby('stock')['amount'] if 'amount' in df.columns else None
        float_group = df.groupby('stock')['float_shares'] if 'float_shares' in df.columns else None
        
        for period in periods:
            # 换手率
            if volume_group is not None and float_group is not None:
                df[f'turnover_{period}w'] = volume_group.transform(lambda s: s.rolling(period).mean()) / float_group.transform(lambda s: s.rolling(period).mean())
            
            # 成交额变化
            if amount_group is not None:
                df[f'amt_chg_{period}w'] = amount_group.pct_change(period)
        
        # 成交量趋势
        if volume_group is not None:
            df['volume_trend'] = volume_group.transform(lambda s: s.rolling(5).mean()) / volume_group.transform(lambda s: s.rolling(20).mean())
        
        # 流动性比率
        if 'amount' in df.columns and 'market_cap' in df.columns:
            df['liquidity_ratio'] = df['amount'] / df['market_cap']
        
        logger.info(f"流动性特征计算完成，特征数: {len(periods) * 2 + 2}")
        
        return df
    
    def calculate_stability_features(
        self,
        df: pd.DataFrame,
        periods: List[int] = None
    ) -> pd.DataFrame:
        """
        计算稳定性/低波动特征（豆包低波动因子，权重20%）

        低波动因子包括：
        - volatility: 日度波动率
        - max_dd: 最大回撤
        - downside_vol: 下行波动率（只统计下跌日的波动）

        Args:
            df: 包含收盘价的DataFrame
            periods: 周期列表

        Returns:
            包含稳定性特征的DataFrame
        """
        periods = periods or self.vol_periods
        df = df.copy()
        df = df.sort_values(['stock', 'date'])
        close_group = df.groupby('stock')['close']

        # 计算日度收益率
        df['daily_ret'] = close_group.pct_change()

        for period in periods:
            # 1. 波动率（年化）
            df[f'volatility_{period}d'] = df.groupby('stock')['daily_ret'].transform(
                lambda s: s.rolling(period).std() * np.sqrt(252)
            )

            # 2. 最大回撤
            rolling_max = close_group.transform(lambda s: s.rolling(period).max())
            df[f'max_dd_{period}d'] = (rolling_max - df['close']) / rolling_max

            # 3. 下行波动率（豆包低波动因子核心）
            # 只统计下跌日的波动，衡量下行风险
            downside_ret = df['daily_ret'].where(df['daily_ret'] < 0, 0)
            df[f'downside_vol_{period}d'] = downside_ret.groupby(df['stock']).transform(
                lambda s: s.rolling(period).std() * np.sqrt(252)
            )

        # 低波动因子综合评分（波动越低，评分越高）
        # 使用 20 日周期作为默认
        vol_col = 'volatility_20d' if 'volatility_20d' in df.columns else f'volatility_{periods[0]}d'
        dd_col = 'max_dd_20d' if 'max_dd_20d' in df.columns else f'max_dd_{periods[0]}d'
        dvol_col = 'downside_vol_20d' if 'downside_vol_20d' in df.columns else f'downside_vol_{periods[0]}d'

        # 标准化（取负值，波动越低评分越高）
        if vol_col in df.columns:
            df['volatility_zscore'] = -df.groupby('stock')[vol_col].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / 
                         (x.rolling(252, min_periods=60).std() + 1e-8)
            )
        if dd_col in df.columns:
            df['max_dd_zscore'] = -df.groupby('stock')[dd_col].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / 
                         (x.rolling(252, min_periods=60).std() + 1e-8)
            )
        if dvol_col in df.columns:
            df['downside_vol_zscore'] = -df.groupby('stock')[dvol_col].transform(
                lambda x: (x - x.rolling(252, min_periods=60).mean()) / 
                         (x.rolling(252, min_periods=60).std() + 1e-8)
            )

        # 低波动因子综合评分
        low_vol_components = []
        for col in ['volatility_zscore', 'max_dd_zscore', 'downside_vol_zscore']:
            if col in df.columns:
                low_vol_components.append(col)

        if low_vol_components:
            # 豆包方案：波动率、最大回撤、下行波动率等权重
            df['low_vol_score'] = df[low_vol_components].mean(axis=1)
        else:
            df['low_vol_score'] = np.nan

        logger.info(f"稳定性特征计算完成，特征数: {len(periods) * 3 + len(low_vol_components) + 1}")

        return df
    
    def calculate_technical_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算技术指标特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含技术指标的DataFrame
        """
        df = df.copy()
        df = df.sort_values(['stock', 'date'])
        close_group = df.groupby('stock')['close']
        
        # 均线斜率
        ma_20 = close_group.transform(lambda s: s.rolling(20).mean())
        ma_60 = close_group.transform(lambda s: s.rolling(60).mean())
        df['ma_slope'] = (ma_20 - ma_60) / 40
        
        # RSI
        df['rsi'] = close_group.transform(lambda s: self._calculate_rsi(s, period=14))
        
        logger.info("技术指标特征计算完成，特征数: 2")
        
        return df
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """
        计算RSI
        
        Args:
            close: 收盘价序列
            period: 周期
            
        Returns:
            RSI序列
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_all_features(
        self,
        df: pd.DataFrame,
        industry_data: Optional[pd.DataFrame] = None,
        fundamental_scores: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算所有价格特征（豆包 Champion 策略四因子体系）

        Args:
            df: 包含OHLCV数据的DataFrame
            industry_data: 行业指数数据（用于计算相对超额收益）
            fundamental_scores: 基本面评分数据（quality_score, growth_score）

        Returns:
            包含所有特征的DataFrame
        """
        logger.info("开始计算价格特征（豆包四因子体系）...")

        # 检查必要的列
        required_cols = ['date', 'stock', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")

        # 计算各类特征
        df = self.calculate_momentum_features(df, industry_data=industry_data)
        df = self.calculate_liquidity_features(df)
        df = self.calculate_stability_features(df)
        df = self.calculate_technical_features(df)

        # 合并基本面评分（如果有）
        if fundamental_scores is not None and not fundamental_scores.empty:
            merge_cols = ['ts_code', 'date']
            available_cols = [c for c in merge_cols if c in fundamental_scores.columns]
            if len(available_cols) >= 2:
                score_cols = ['ts_code', 'date', 'quality_score', 'growth_score', 'fundamental_score']
                score_cols = [c for c in score_cols if c in fundamental_scores.columns]
                if len(score_cols) >= 2:
                    df = df.merge(fundamental_scores[score_cols], on=available_cols, how='left')

        # 计算豆包四因子综合评分
        # 质量（30%）+ 成长（30%）+ 动量（20%）+ 低波动（20%）
        score_components = []
        for col in ['quality_score', 'growth_score', 'momentum_score', 'low_vol_score']:
            if col in df.columns:
                score_components.append(col)

        if score_components:
            df['doubao_champion_score'] = (
                df.get('quality_score', 0) * DOUBAO_FACTOR_WEIGHTS['quality'] +
                df.get('growth_score', 0) * DOUBAO_FACTOR_WEIGHTS['growth'] +
                df.get('momentum_score', 0) * DOUBAO_FACTOR_WEIGHTS['momentum'] +
                df.get('low_vol_score', 0) * DOUBAO_FACTOR_WEIGHTS['low_vol']
            )

        logger.info(f"价格特征计算完成，总特征数: {len(df.columns) - len(required_cols)}")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        return self.calculate_all_features(df)

    def get_feature_names(self) -> List[str]:
        """返回生成的特征名称列表（豆包四因子体系）"""
        return [
            # 动量因子（20%）
            'momentum_20d', 'momentum_60d',
            'excess_return_vs_industry_20d', 'excess_return_vs_industry_60d',
            'excess_return_vs_industry_20d_zscore', 'excess_return_vs_industry_60d_zscore',
            'amt_quantile', 'amt_ratio_20_250', 'amt_quantile_zscore',
            'momentum_score',
            # 低波动因子（20%）
            'volatility_20d', 'volatility_60d',
            'max_dd_20d', 'max_dd_60d',
            'downside_vol_20d', 'downside_vol_60d',
            'volatility_zscore', 'max_dd_zscore', 'downside_vol_zscore',
            'low_vol_score',
            # 流动性特征
            'turnover_4w', 'turnover_12w',
            'amt_chg_4w', 'amt_chg_12w',
            'volume_trend', 'liquidity_ratio',
            # 技术指标
            'ma_slope', 'rsi',
            # 综合评分
            'doubao_champion_score',
        ]


if __name__ == "__main__":
    # 测试价格特征提取
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
    stocks = ['sh.600000', 'sh.600001']
    
    data = []
    for stock in stocks:
        close = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'stock': stock,
                'open': close[i] * (1 + np.random.randn() * 0.01),
                'high': close[i] * (1 + abs(np.random.randn() * 0.02)),
                'low': close[i] * (1 - abs(np.random.randn() * 0.02)),
                'close': close[i],
                'volume': np.random.randint(1000000, 10000000),
                'amount': close[i] * np.random.randint(1000000, 10000000),
                'float_shares': 100000000,
                'market_cap': close[i] * 100000000
            })
    
    df = pd.DataFrame(data)
    
    feature_extractor = PriceFeatures()
    df_with_features = feature_extractor.calculate_all_features(df)
    
    print(f"\n特征列:")
    print(df_with_features.columns.tolist())
    
    print(f"\n数据预览:")
    print(df_with_features.head())
