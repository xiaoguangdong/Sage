"""
买卖点过滤模型（Logistic Regression）
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EntryModelLR:
    """Logistic回归买卖点模型"""
    
    def __init__(self, config: dict):
        """
        初始化买卖点模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
        # 模型参数
        self.model_params = {
            'max_iter': 1000,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        # 规则参数
        self.rules = {
            'prob_threshold': 0.6,
            'ma_period': 20,
            'vol_threshold': 0.05
        }
        
        if config:
            if 'model_params' in config:
                self.model_params.update(config['model_params'])
            if 'rules' in config:
                self.rules.update(config['rules'])
        
        # 标签参数
        self.label_params = {
            'horizon': 10,
            'return_threshold': 0.05,
            'max_drawdown': -0.03
        }
        
        if config and 'label' in config:
            self.label_params.update(config['label'])
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            包含特征的DataFrame
        """
        df = df.copy()
        
        # 价格位置特征
        for period in [5, 10, 20, 60]:
            df[f'price_ma{period}_ratio'] = df['close'] / df['close'].rolling(period).mean()
            df[f'price_std{period}_ratio'] = (df['close'] - df['close'].rolling(period).mean()) / df['close'].rolling(period).std()
        
        # 成交量分析特征
        for period in [5, 10, 20]:
            df[f'volume_ma{period}_ratio'] = df['turnover'] / df['turnover'].rolling(period).mean()
            df[f'volume_price_corr{period}'] = df['close'].rolling(period).corr(df['turnover'])
        
        # MACD特征
        macd, macd_signal = self.calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd - macd_signal
        df['macd_signal_cross'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        
        # K线形态特征
        df['is_doji'] = self.is_doji(df)
        df['is_hammer'] = self.is_hammer(df)
        df['is_shooting_star'] = self.is_shooting_star(df)
        
        # 波动率特征
        df['volatility_5d'] = df['close'].pct_change().rolling(5).std()
        df['volatility_10d'] = df['close'].pct_change().rolling(10).std()
        
        # 趋势强度特征
        df['trend_strength_20'] = self.calculate_trend_strength(df['close'], 20)
        
        # 去除NaN
        df = df.dropna()
        
        return df
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
    
    def is_doji(self, df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """判断是否为十字星"""
        if 'high' not in df.columns or 'low' not in df.columns:
            return pd.Series(0, index=df.index)
        
        body = abs(df['close'] - df['close'].shift(1))
        range_ = df['high'] - df['low']
        doji = (body / range_ < threshold).astype(int)
        return doji
    
    def is_hammer(self, df: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
        """判断是否为锤子线"""
        if 'high' not in df.columns or 'low' not in df.columns:
            return pd.Series(0, index=df.index)
        
        body = abs(df['close'] - df['close'].shift(1))
        lower_shadow = df['close'].shift(1) - df['low']
        upper_shadow = df['high'] - df['close'].shift(1)
        hammer = ((lower_shadow > threshold * body) & (upper_shadow < 0.5 * body)).astype(int)
        return hammer
    
    def is_shooting_star(self, df: pd.DataFrame, threshold: float = 2.0) -> pd.Series:
        """判断是否为射击之星"""
        if 'high' not in df.columns or 'low' not in df.columns:
            return pd.Series(0, index=df.index)
        
        body = abs(df['close'] - df['close'].shift(1))
        upper_shadow = df['high'] - df['close'].shift(1)
        lower_shadow = df['close'].shift(1) - df['low']
        shooting_star = ((upper_shadow > threshold * body) & (lower_shadow < 0.5 * body)).astype(int)
        return shooting_star
    
    def calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """计算趋势强度"""
        return prices.rolling(period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean() if len(x) > 0 else 0
        )
    
    def create_entry_label(self, df: pd.DataFrame) -> pd.Series:
        """
        创建买卖点标签
        
        规则：在horizon天内，如果最大收益率超过threshold，且最大回撤小于max_drawdown，则label=1
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            买卖点标签
        """
        horizon = self.label_params['horizon']
        return_threshold = self.label_params['return_threshold']
        max_drawdown = self.label_params['max_drawdown']
        
        # 计算未来N天的累积收益率
        future_returns = df['close'].shift(-horizon) / df['close'] - 1
        
        # 计算未来N天的最大回撤
        future_prices = df['close'].shift(-1).rolling(window=horizon, min_periods=1).max()
        drawdowns = (df['close'] - future_prices) / df['close']
        max_dd = drawdowns.rolling(window=horizon, min_periods=1).min()
        
        # 创建标签：最大收益率超过阈值且最大回撤小于限制
        labels = ((future_returns >= return_threshold) & (max_dd >= max_drawdown)).astype(int)
        
        return labels
    
    def train(self, df: pd.DataFrame, labels: pd.Series):
        """
        训练模型
        
        Args:
            df: 训练数据DataFrame
            labels: 标签
        """
        logger.info("开始训练买卖点模型...")
        
        # 准备特征
        df_features = self.prepare_features(df)
        if df_features.empty:
            logger.warning("训练数据特征为空，跳过训练")
            self.is_trained = False
            return
        
        # 获取特征名称
        feature_cols = [col for col in df_features.columns if col not in ['date', 'code', 'close', 'turnover']]
        self.feature_names = feature_cols
        
        # 对齐数据
        common_index = df_features.index.intersection(labels.index)
        X = df_features.loc[common_index, feature_cols].values
        y = labels.loc[common_index].values
        
        if len(np.unique(y)) < 2:
            logger.warning("训练标签仅包含单一类别，使用常量概率模型")
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            self.model = None
            self.is_trained = True
            self.constant_prob = float(y[0]) if len(y) else 0.0
            return

        # 标准化特征
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 训练模型
        self.model = LogisticRegression(**self.model_params)
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        logger.info(f"买卖点模型训练完成，训练集样本数: {len(X)}")
        logger.info(f"正样本比例: {y.mean():.2%}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预测买卖点
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            包含预测结果的DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 准备特征
        df_features = self.prepare_features(df)
        if df_features.empty:
            df_result = df_features.copy()
            df_result['entry_prob'] = []
            df_result['above_ma'] = []
            df_result['low_vol'] = []
            df_result['entry_signal'] = []
            return df_result
        
        # 获取特征名称
        feature_cols = self.feature_names
        
        # 标准化特征
        X = df_features[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # 预测概率
        if self.model is None:
            prob = np.full(len(X_scaled), self.constant_prob, dtype=float)
        else:
            prob = self.model.predict_proba(X_scaled)[:, 1]
        
        # 应用规则过滤
        ma_period = self.rules['ma_period']
        vol_threshold = self.rules['vol_threshold']
        prob_threshold = self.rules['prob_threshold']
        
        # 规则1: 价格高于MA20
        above_ma = df_features['close'] > df_features['close'].rolling(ma_period).mean()
        
        # 规则2: 波动率低于阈值
        volatility = df_features['close'].pct_change().rolling(ma_period).std()
        low_vol = volatility < vol_threshold
        
        # 规则3: 模型概率高于阈值
        high_prob = prob >= prob_threshold
        
        # 最终判断
        entry_signal = high_prob & above_ma & low_vol
        
        # 添加预测结果
        df_result = df_features.copy()
        df_result['entry_prob'] = prob
        df_result['above_ma'] = above_ma
        df_result['low_vol'] = low_vol
        df_result['entry_signal'] = entry_signal.astype(int)
        
        return df_result
    
    def predict_batch(
        self,
        stock_histories: Dict[str, pd.DataFrame],
        existing_holdings: Optional[set] = None,
    ) -> Dict[str, bool]:
        """
        批量预测多只股票的买入信号

        Args:
            stock_histories: {ts_code: DataFrame}，每只股票的历史日线数据（需含 close, turnover）
            existing_holdings: 已持有的股票集合，这些股票默认放行

        Returns:
            {ts_code: bool}，True 表示有买入信号
        """
        if existing_holdings is None:
            existing_holdings = set()

        if not self.is_trained:
            # 未训练时全部放行
            return {ts_code: True for ts_code in stock_histories}

        results = {}
        for ts_code, hist_df in stock_histories.items():
            # 已持有的股票直接放行
            if ts_code in existing_holdings:
                results[ts_code] = True
                continue

            try:
                if len(hist_df) < 30:
                    # 历史数据不足，放行
                    results[ts_code] = True
                    continue

                pred = self.predict(hist_df)
                if pred.empty:
                    results[ts_code] = True
                    continue

                # 取最后一天的信号
                results[ts_code] = bool(pred['entry_signal'].iloc[-1] == 1)
            except Exception as e:
                logger.debug(f"EntryModel predict failed for {ts_code}: {e}")
                results[ts_code] = True  # 异常时放行

        return results

    def get_model_params(self) -> Dict:
        """
        获取模型参数
        
        Returns:
            模型参数字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        params = {
            'intercept': self.model.intercept_[0],
            'feature_importance': dict(zip(self.feature_names, self.model.coef_[0]))
        }
        
        return params


if __name__ == "__main__":
    # 测试买卖点模型
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
    
    # 模拟股票数据
    close = 10 + np.cumsum(np.random.randn(len(dates)) * 0.1)
    turnover = np.random.uniform(0.01, 0.1, len(dates))
    
    data = []
    for i, date in enumerate(dates):
        data.append({
            'date': date,
            'code': 'sh.600000',
            'close': close[i],
            'turnover': turnover[i]
        })
    
    df = pd.DataFrame(data)
    
    # 测试模型
    print("测试买卖点模型...")
    config = {
        'rules': {
            'prob_threshold': 0.6,
            'ma_period': 20,
            'vol_threshold': 0.05
        },
        'label': {
            'horizon': 10,
            'return_threshold': 0.05,
            'max_drawdown': -0.03
        }
    }
    
    model = EntryModelLR(config)
    
    # 创建标签
    labels = model.create_entry_label(df)
    df_clean = df.dropna()
    labels = labels.loc[df_clean.index]
    
    # 训练模型
    model.train(df_clean, labels)
    
    # 预测
    df_predict = model.predict(df.tail(30))
    
    print(f"\n预测结果（最后30条）:")
    print(df_predict[['date', 'code', 'entry_prob', 'above_ma', 'low_vol', 'entry_signal']].head(10))
    
    # 买入信号数量
    signal_count = df_predict['entry_signal'].sum()
    print(f"\n买入信号数量: {signal_count}/{len(df_predict)}")
    
    # 模型参数
    params = model.get_model_params()
    print(f"\n模型截距: {params['intercept']:.4f}")
    print(f"特征重要性（Top 5）:")
    importance = sorted(params['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    for feature, imp in importance:
        print(f"  {feature}: {imp:.4f}")
