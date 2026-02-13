"""
市场特征（指数级特征）
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

from .base import FeatureGenerator, FeatureSpec
from .registry import register_feature

logger = logging.getLogger(__name__)


@register_feature
class MarketFeatures(FeatureGenerator):
    """市场特征提取器（基于沪深300指数）"""

    spec = FeatureSpec(
        name="market_features",
        input_fields=("date", "close"),
        description="指数级市场特征（趋势/波动/量价/回撤）",
    )
    
    def __init__(self, index_code: str = "000300.SH"):
        """
        初始化市场特征提取器
        
        Args:
            index_code: 指数代码
        """
        self.index_code = index_code
        
    def calculate_basic_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        计算基础特征（8个）
        
        Args:
            df: 包含指数数据的DataFrame
            
        Returns:
            包含基础特征的DataFrame
        """
        df = df.copy()
        
        # 收益指标
        df['ret_4w'] = df['close'].pct_change(4)
        df['ret_12w'] = df['close'].pct_change(12)
        
        # 波动率指标
        df['vol_4w'] = df['close'].pct_change().rolling(4).std()
        df['vol_12w'] = df['close'].pct_change().rolling(12).std()
        
        # 成交额变化
        if 'amount' in df.columns:
            df['amt_chg'] = df['amount'].pct_change(4)
        else:
            df['amt_chg'] = np.nan
        
        # 均线指标
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_60'] = df['close'].rolling(60).mean()
        df['ma_120'] = df['close'].rolling(120).mean()
        
        logger.info("基础特征计算完成，特征数: 8")
        
        return df
    
    def calculate_price_rank_features(
        self,
        df: pd.DataFrame,
        windows: List[int] = [20, 60]
    ) -> pd.DataFrame:
        """
        计算价格排名特征
        
        Args:
            df: 包含指数数据的DataFrame
            windows: 窗口列表
            
        Returns:
            包含价格排名特征的DataFrame
        """
        df = df.copy()
        
        for window in windows:
            df[f'price_rank_{window}d'] = df['close'].rolling(window).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1],
                raw=False
            )
        
        logger.info(f"价格排名特征计算完成，特征数: {len(windows)}")
        
        return df
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋势强度
        
        Args:
            df: 包含指数数据的DataFrame
            
        Returns:
            包含趋势强度的DataFrame
        """
        df = df.copy()
        
        ma_20 = df['ma_20']
        ma_60 = df['ma_60']
        
        df['trend_strength'] = (ma_20 - ma_60) / ma_60
        
        logger.info("趋势强度计算完成")
        
        return df
    
    def calculate_volatility_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算波动率比率
        
        Args:
            df: 包含指数数据的DataFrame
            
        Returns:
            包含波动率比率的DataFrame
        """
        df = df.copy()
        
        df['vol_ratio'] = df['vol_4w'] / df['vol_12w']
        
        logger.info("波动率比率计算完成")
        
        return df
    
    def calculate_extreme_vol_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算极端波动次数
        
        Args:
            df: 包含指数数据的DataFrame
            
        Returns:
            包含极端波动次数的DataFrame
        """
        df = df.copy()
        
        # 统计20天内有多少次极端波动（超过2倍标准差）
        df['extreme_vol_count'] = (abs(df['close'].pct_change()) > 2 * df['vol_12w']).rolling(20).sum()
        
        logger.info("极端波动次数计算完成")
        
        return df
    
    def calculate_amt_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算成交额趋势
        
        Args:
            df: 包含指数数据的DataFrame
            
        Returns:
            包含成交额趋势的DataFrame
        """
        df = df.copy()
        
        if 'amount' in df.columns:
            df['amt_trend_20d'] = df['amount'].rolling(20).mean() / df['amount'].rolling(60).mean()
        else:
            df['amt_trend_20d'] = np.nan
        
        logger.info("成交额趋势计算完成")
        
        return df
    
    def calculate_price_volume_corr(self, df: pd.DataFrame, period: int = 4) -> pd.DataFrame:
        """
        计算量价相关性
        
        Args:
            df: 包含指数数据的DataFrame
            period: 周期
            
        Returns:
            包含量价相关性的DataFrame
        """
        df = df.copy()
        
        if 'amount' in df.columns:
            df['price_volume_corr'] = df['close'].pct_change(period).rolling(period).corr(
                df['amount'].pct_change(period)
            )
        else:
            df['price_volume_corr'] = np.nan
        
        logger.info("量价相关性计算完成")
        
        return df
    
    def calculate_max_drawdown(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        计算最大回撤
        
        Args:
            df: 包含指数数据的DataFrame
            window: 窗口
            
        Returns:
            包含最大回撤的DataFrame
        """
        df = df.copy()
        
        rolling_max = df['close'].rolling(window).max()
        df[f'max_dd_{window}d'] = (rolling_max - df['close']) / rolling_max
        
        logger.info("最大回撤计算完成")
        
        return df
    
    def calculate_drawdown_count(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        计算回撤频率
        
        Args:
            df: 包含指数数据的DataFrame
            window: 窗口
            
        Returns:
            包含回撤频率的DataFrame
        """
        df = df.copy()
        
        # 统计20天内有多少次回撤超过5%
        df['drawdown_count'] = (df['close'] < df['close'].rolling(window).max() * 0.95).rolling(window).sum()
        
        logger.info("回撤频率计算完成")
        
        return df
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        计算ATR（真实波动幅度）
        
        Args:
            df: 包含OHLC数据的DataFrame
            period: 周期
            
        Returns:
            包含ATR的DataFrame
        """
        df = df.copy()
        
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        # 计算True Range
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        
        logger.info("ATR计算完成")
        
        return df
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有市场特征（基于沪深300指数）
        
        Args:
            df: 包含指数OHLCV数据的DataFrame
            
        Returns:
            包含所有特征的DataFrame
        """
        logger.info(f"开始计算{self.index_code}的市场特征...")
        
        # 检查必要的列
        required_cols = ['date', 'close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 计算基础特征（8个）
        df = self.calculate_basic_features(df)
        
        # 计算硬核指标（10个）
        df = self.calculate_price_rank_features(df, windows=[20, 60])
        df = self.calculate_trend_strength(df)
        df = self.calculate_volatility_ratio(df)
        df = self.calculate_extreme_vol_count(df)
        df = self.calculate_amt_trend(df)
        df = self.calculate_price_volume_corr(df)
        df = self.calculate_max_drawdown(df)
        df = self.calculate_drawdown_count(df)
        df = self.calculate_atr(df)
        
        logger.info(f"市场特征计算完成，总特征数: {len(df.columns) - len(required_cols)}")
        
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)
        return self.calculate_all_features(df)


if __name__ == "__main__":
    # 测试市场特征提取
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
    
    data = []
    close_series = 3000 + np.cumsum(np.random.randn(len(dates)) * 10)
    for i, date in enumerate(dates):
        close = close_series[i]
        data.append({
            'date': date,
            'close': close,
            'high': close * (1 + abs(np.random.randn() * 0.01)),
            'low': close * (1 - abs(np.random.randn() * 0.01)),
            'amount': close * 1e8 * np.random.uniform(0.8, 1.2)
        })
    
    df = pd.DataFrame(data)
    
    feature_extractor = MarketFeatures("000300.SH")
    df_with_features = feature_extractor.calculate_all_features(df)
    
    print(f"\n特征列:")
    print(df_with_features.columns.tolist())
    
    print(f"\n数据预览:")
    print(df_with_features.head())
