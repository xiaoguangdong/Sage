"""
选股排序模型（LightGBM Ranker）
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RankModelLGBM:
    """LightGBM排序模型"""
    
    def __init__(self, config: dict):
        """
        初始化排序模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        # LightGBM参数
        self.lgbm_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'num_leaves': 16,
            'max_depth': 4,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        if config and 'lgbm_params' in config:
            self.lgbm_params.update(config['lgbm_params'])
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            包含特征的DataFrame
        """
        df = df.copy()
        
        # 动量特征
        for period in [4, 12, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)
            df[f'ma_ratio_{period}d'] = df['close'] / df['close'].rolling(period).mean()
        
        # 流动性特征
        for period in [4, 12]:
            df[f'turnover_{period}d_mean'] = df['turnover'].rolling(period).mean()
            df[f'turnover_{period}d_std'] = df['turnover'].rolling(period).std()
            df[f'turnover_ratio_{period}d'] = df['turnover'] / df['turnover'].rolling(period).mean()
        
        # 稳定性特征
        for period in [4, 12]:
            df[f'volatility_{period}d'] = df['close'].pct_change().rolling(period).std()
            df[f'price_std_{period}d'] = df['close'].rolling(period).std()
        
        # 技术指标
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'] = self.calculate_macd(df['close'])
        df['bollinger_upper'], df['bollinger_lower'] = self.calculate_bollinger(df['close'], 20)
        
        # 去除NaN
        df = df.dropna()
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
    
    def calculate_bollinger(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
        """计算布林带"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def create_ranking_label(self, df: pd.DataFrame, horizon: int = 20) -> pd.Series:
        """
        创建排序标签（未来N天收益率排名）
        
        Args:
            df: 股票数据DataFrame
            horizon: 预测天数
            
        Returns:
            排序标签
        """
        # 计算未来N天收益率
        future_returns = df['close'].shift(-horizon) / df['close'] - 1
        
        # 计算相对排名（0-1之间）
        rank_label = future_returns.rank(pct=True)
        
        return rank_label
    
    def train(self, df: pd.DataFrame, labels: pd.Series, group_info: pd.Series = None):
        """
        训练模型
        
        Args:
            df: 训练数据DataFrame
            labels: 标签
            group_info: 分组信息（按日期分组）
        """
        logger.info("开始训练排序模型...")
        
        # 准备特征
        df_features = self.prepare_features(df)
        
        # 获取特征名称（防数据泄露：过滤label/future字段）
        exclude_cols = {'date', 'trade_date', 'code', 'ts_code', 'stock', 'close', 'turnover'}
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]

        leakage_cols = [col for col in feature_cols if 'label' in col.lower() or 'future' in col.lower()]
        label_name = labels.name if labels is not None else None
        if label_name and label_name in feature_cols:
            leakage_cols.append(label_name)

        if leakage_cols:
            logger.warning(f"疑似数据泄露字段已剔除: {sorted(set(leakage_cols))}")
            feature_cols = [col for col in feature_cols if col not in set(leakage_cols)]

        self.feature_names = feature_cols
        
        # 创建数据集
        train_data = lgb.Dataset(
            df_features[feature_cols].values,
            label=labels.values,
            group=group_info.values if group_info is not None else None
        )
        
        # 训练模型
        self.model = lgb.train(
            self.lgbm_params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(period=10)]
        )
        
        self.is_trained = True
        logger.info("排序模型训练完成")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测股票排序
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            包含预测分数的DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 准备特征
        df_features = self.prepare_features(df)
        
        # 获取特征名称
        feature_cols = self.feature_names
        
        # 预测
        scores = self.model.predict(df_features[feature_cols].values)
        
        # 添加预测分数
        df_result = df_features.copy()
        df_result['rank_score'] = scores
        df_result['rank'] = df_result['rank_score'].rank(ascending=False)
        
        return df_result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            特征重要性DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        importance = self.model.feature_importance()
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance


if __name__ == "__main__":
    # 测试排序模型
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
    
    # 模拟多只股票的数据
    stock_codes = ['sh.600000', 'sh.600004', 'sh.600006']
    all_data = []
    
    for code in stock_codes:
        close = 10 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        turnover = np.random.uniform(0.01, 0.1, len(dates))
        
        for i, date in enumerate(dates):
            all_data.append({
                'date': date,
                'code': code,
                'close': close[i],
                'turnover': turnover[i]
            })
    
    df = pd.DataFrame(all_data)
    
    # 测试模型
    print("测试排序模型...")
    config = {
        'lgbm_params': {
            'num_leaves': 16,
            'max_depth': 4
        }
    }
    
    model = RankModelLGBM(config)
    
    # 创建标签
    labels = model.create_ranking_label(df)
    df = df.dropna()
    labels = labels.loc[df.index]
    
    # 创建分组信息
    group_info = df.groupby('date').size()
    
    # 训练模型
    model.train(df, labels, group_info)
    
    # 预测
    df_predict = model.predict(df.tail(30))
    
    print(f"\n预测结果（最后30条）:")
    print(df_predict[['date', 'code', 'rank_score', 'rank']].head(10))
    
    # 特征重要性
    importance = model.get_feature_importance()
    print(f"\n特征重要性（Top 10）:")
    print(importance.head(10))
