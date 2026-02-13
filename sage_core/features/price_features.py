"""
价格特征（动量、波动、量价）
"""
import pandas as pd
import numpy as np
from typing import Optional, List
import logging

from .base import FeatureGenerator, FeatureSpec
from .registry import register_feature

logger = logging.getLogger(__name__)


@register_feature
class PriceFeatures(FeatureGenerator):
    """价格特征提取器"""

    spec = FeatureSpec(
        name="price_features",
        input_fields=("date", "stock", "close"),
        description="个股价格相关特征（动量/流动性/波动/技术指标）",
    )
    
    def __init__(self):
        """初始化价格特征提取器"""
        pass
    
    def calculate_momentum_features(
        self,
        df: pd.DataFrame,
        periods: List[int] = [4, 12, 20]
    ) -> pd.DataFrame:
        """
        计算动量特征
        
        Args:
            df: 包含收盘价的DataFrame
            periods: 周期列表
            
        Returns:
            包含动量特征的DataFrame
        """
        df = df.copy()
        df = df.sort_values(['stock', 'date'])
        close_group = df.groupby('stock')['close']
        
        for period in periods:
            # 价格动量
            df[f'mom_{period}w'] = close_group.pct_change(period)
            
            # 相对强度vs指数（如果有指数数据）
            # 这里暂时用简单版本，后续需要传入指数数据
            df[f'rs_{period}w'] = df[f'mom_{period}w']  # 后续需要改为相对指数
        
        # 动量加速度
        df['mom_acceleration'] = df['mom_4w'] - df['mom_8w'] if 'mom_8w' in df.columns else np.nan
        
        # 相对强度
        df['relative_strength'] = close_group.transform(lambda s: s.rolling(60).mean()) / close_group.shift(1) - 1
        
        logger.info(f"动量特征计算完成，特征数: {len(periods) * 2 + 2}")
        
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
        periods: List[int] = [4, 12]
    ) -> pd.DataFrame:
        """
        计算稳定性特征
        
        Args:
            df: 包含收盘价的DataFrame
            periods: 周期列表
            
        Returns:
            包含稳定性特征的DataFrame
        """
        df = df.copy()
        df = df.sort_values(['stock', 'date'])
        close_group = df.groupby('stock')['close']
        
        for period in periods:
            # 波动率
            df[f'vol_{period}w'] = close_group.transform(lambda s: s.pct_change().rolling(period).std())
            
            # 最大回撤
            rolling_max = close_group.transform(lambda s: s.rolling(period).max())
            df[f'max_dd_{period}w'] = (rolling_max - df['close']) / rolling_max
        
        # 价格稳定性
        if 'vol_4w' in df.columns:
            df['price_stability'] = df['vol_4w'] / close_group.transform(lambda s: s.rolling(4).mean())
        
        logger.info(f"稳定性特征计算完成，特征数: {len(periods) * 2 + 1}")
        
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
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算所有价格特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            包含所有特征的DataFrame
        """
        logger.info("开始计算价格特征...")
        
        # 检查必要的列
        required_cols = ['date', 'stock', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 计算各类特征
        df = self.calculate_momentum_features(df)
        df = self.calculate_liquidity_features(df)
        df = self.calculate_stability_features(df)
        df = self.calculate_technical_features(df)
        
        logger.info(f"价格特征计算完成，总特征数: {len(df.columns) - len(required_cols)}")
        
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        return self.calculate_all_features(df)


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
