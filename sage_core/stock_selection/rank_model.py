"""
选股排序模型（LightGBM Ranker）
"""

import logging
from typing import Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from sage_core.features import FlowFeatures, FundamentalFeatures, IndustryFeatures, PriceFeatures

logger = logging.getLogger(__name__)


class RankModelLGBM:
    """LightGBM排序模型"""

    # 默认特征预算（符合设计文档 20~30 个）
    DEFAULT_FEATURE_BUDGET = {
        "quality": 6,  # 质量：roe, gross_margin, roic, quality_score
        "momentum": 8,  # 动量：return_*, ma_ratio_*, rsi, macd
        "flow": 4,  # 资金：northbound_*, main_flow_*
        "risk": 6,  # 风险：volatility_*, price_std_*, bollinger_*
        "valuation": 4,  # 估值：pe_*, pb_*
        "industry": 4,  # 行业：industry_ret_*, industry_momentum_score
    }

    def __init__(self, config: dict = None):
        """
        初始化排序模型

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.model = None
        self.feature_names = None
        self.is_trained = False

        # 特征提取器
        self.price_features = PriceFeatures()
        self.fundamental_features = FundamentalFeatures()
        self.flow_features = FlowFeatures()
        self.industry_features = IndustryFeatures()

        # 特征预算（可配置覆盖）
        self.feature_budget = self.config.get("feature_budget", self.DEFAULT_FEATURE_BUDGET)

        # LightGBM参数
        self.lgbm_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "num_leaves": 16,
            "max_depth": 4,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

        if self.config and "lgbm_params" in self.config:
            self.lgbm_params.update(self.config["lgbm_params"])

    def prepare_features(
        self,
        df: pd.DataFrame,
        daily_basic: Optional[pd.DataFrame] = None,
        fina_indicator: Optional[pd.DataFrame] = None,
        hk_hold: Optional[pd.DataFrame] = None,
        moneyflow: Optional[pd.DataFrame] = None,
        industry_index: Optional[pd.DataFrame] = None,
        industry_col: str = "industry_l1",
    ) -> pd.DataFrame:
        """
        准备特征（集成多源特征模块）

        Args:
            df: 股票数据DataFrame（需包含 ts_code, date, close, turnover）
            daily_basic: 每日基本面数据（pe_ttm, pb 等）
            fina_indicator: 财务指标数据（roe, gross_margin, roic 等）
            hk_hold: 港资持股数据
            moneyflow: 资金流向数据
            industry_index: 行业指数数据
            industry_col: 行业分类字段名

        Returns:
            包含特征的DataFrame
        """
        df = df.copy()

        # 标准化字段名
        if "trade_date" in df.columns and "date" not in df.columns:
            df["date"] = df["trade_date"]
        if "code" in df.columns and "ts_code" not in df.columns:
            df["ts_code"] = df["code"]

        # 1. 价格/动量/流动性/技术指标特征
        df = self._calculate_price_features(df)

        # 2. 估值/质量特征
        if daily_basic is not None or fina_indicator is not None:
            df = self.fundamental_features.calculate_all_features(df, daily_basic, fina_indicator)

        # 3. 资金流向特征
        if hk_hold is not None or moneyflow is not None:
            df = self.flow_features.calculate_all_features(df, hk_hold=hk_hold, moneyflow=moneyflow)

        # 4. 行业动量特征
        if industry_index is not None or industry_col in df.columns:
            df = self.industry_features.calculate_all_features(
                df, industry_index=industry_index, industry_col=industry_col
            )

        # 选择特征列（按预算）
        df = self._select_features_by_budget(df)

        return df

    def _calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格相关特征（动量/流动性/波动/技术指标）"""
        df = df.copy()

        # 确保有 turnover 列
        if "turnover" not in df.columns:
            if "turnover_rate" in df.columns:
                df["turnover"] = df["turnover_rate"]
            else:
                df["turnover"] = np.nan

        # 动量特征
        for period in [4, 12, 20]:
            df[f"return_{period}d"] = df["close"].pct_change(period)
            df[f"ma_ratio_{period}d"] = df["close"] / df["close"].rolling(period).mean()

        # 流动性特征
        for period in [4, 12]:
            df[f"turnover_{period}d_mean"] = df["turnover"].rolling(period).mean()
            df[f"turnover_{period}d_std"] = df["turnover"].rolling(period).std()
            df[f"turnover_ratio_{period}d"] = df["turnover"] / df["turnover"].rolling(period).mean()

        # 稳定性特征
        for period in [4, 12]:
            df[f"volatility_{period}d"] = df["close"].pct_change().rolling(period).std()
            df[f"price_std_{period}d"] = df["close"].rolling(period).std()

        # 技术指标
        df["rsi_14"] = self.calculate_rsi(df["close"], 14)
        df["macd"], df["macd_signal"] = self.calculate_macd(df["close"])
        df["bollinger_upper"], df["bollinger_lower"] = self.calculate_bollinger(df["close"], 20)

        return df

    def _select_features_by_budget(self, df: pd.DataFrame) -> pd.DataFrame:
        """按特征预算选择特征列"""
        # 定义特征类别映射
        feature_categories = {
            "quality": [
                "roe",
                "gross_margin",
                "roic",
                "quality_score",
                "roe_zscore",
                "gross_margin_zscore",
                "roic_zscore",
            ],
            "momentum": [
                "return_4d",
                "return_12d",
                "return_20d",
                "ma_ratio_4d",
                "ma_ratio_12d",
                "ma_ratio_20d",
                "rsi_14",
                "macd",
                "macd_signal",
            ],
            "flow": [
                "northbound_hold_ratio",
                "northbound_score",
                "main_net_inflow",
                "main_flow_score",
                "northbound_net_flow_5d_sum",
            ],
            "risk": [
                "volatility_4d",
                "volatility_12d",
                "price_std_4d",
                "price_std_12d",
                "bollinger_upper",
                "bollinger_lower",
            ],
            "valuation": [
                "pe_ttm",
                "pb",
                "pe_percentile",
                "pb_percentile",
                "valuation_score",
            ],
            "industry": [
                "industry_ret_4w",
                "industry_ret_12w",
                "industry_momentum_score",
                "industry_relative_strength",
            ],
        }

        # 选择每个类别中存在且非空的特征
        selected_features = []
        for category, features in feature_categories.items():
            budget = self.feature_budget.get(category, 6)
            available = [f for f in features if f in df.columns and df[f].notna().any()]
            selected_features.extend(available[:budget])

        logger.info(f"按预算选择特征: {len(selected_features)} 个")
        return df

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
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
        future_returns = df["close"].shift(-horizon) / df["close"] - 1

        # 计算相对排名（0-1之间）
        rank_label = future_returns.rank(pct=True)

        return rank_label

    def train(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        group_info: pd.Series = None,
        daily_basic: Optional[pd.DataFrame] = None,
        fina_indicator: Optional[pd.DataFrame] = None,
        hk_hold: Optional[pd.DataFrame] = None,
        moneyflow: Optional[pd.DataFrame] = None,
        industry_index: Optional[pd.DataFrame] = None,
        industry_col: str = "industry_l1",
    ):
        """
        训练模型

        Args:
            df: 训练数据DataFrame
            labels: 标签
            group_info: 分组信息（按日期分组）
            daily_basic: 每日基本面数据
            fina_indicator: 财务指标数据
            hk_hold: 港资持股数据
            moneyflow: 资金流向数据
            industry_index: 行业指数数据
            industry_col: 行业分类字段名
        """
        logger.info("开始训练排序模型...")

        # 准备特征
        df_features = self.prepare_features(
            df,
            daily_basic=daily_basic,
            fina_indicator=fina_indicator,
            hk_hold=hk_hold,
            moneyflow=moneyflow,
            industry_index=industry_index,
            industry_col=industry_col,
        )

        # 获取特征名称（防数据泄露：过滤label/future字段）
        exclude_cols = {
            "date",
            "trade_date",
            "code",
            "ts_code",
            "stock",
            "close",
            "turnover",
            "open",
            "high",
            "low",
            "volume",
            "amount",
            "industry_l1",
            "industry_l2",
            "industry_name",
            "industry_code",
        }
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]

        # 过滤疑似数据泄露字段
        leakage_cols = [col for col in feature_cols if "label" in col.lower() or "future" in col.lower()]
        label_name = labels.name if labels is not None else None
        if label_name and label_name in feature_cols:
            leakage_cols.append(label_name)

        if leakage_cols:
            logger.warning(f"疑似数据泄露字段已剔除: {sorted(set(leakage_cols))}")
            feature_cols = [col for col in feature_cols if col not in set(leakage_cols)]

        # 过滤全为 NaN 的特征
        feature_cols = [col for col in feature_cols if df_features[col].notna().any()]

        self.feature_names = feature_cols
        logger.info(f"最终特征数量: {len(feature_cols)}")

        # 对齐标签索引
        df_features = df_features.dropna(subset=feature_cols)
        if labels is not None:
            labels = labels.loc[df_features.index]

        # 创建数据集
        train_data = lgb.Dataset(
            df_features[feature_cols].values,
            label=labels.values if labels is not None else None,
            group=group_info.values if group_info is not None else None,
        )

        # 训练模型
        self.model = lgb.train(
            self.lgbm_params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(period=10)],
        )

        self.is_trained = True
        logger.info("排序模型训练完成")

    def predict(
        self,
        df: pd.DataFrame,
        daily_basic: Optional[pd.DataFrame] = None,
        fina_indicator: Optional[pd.DataFrame] = None,
        hk_hold: Optional[pd.DataFrame] = None,
        moneyflow: Optional[pd.DataFrame] = None,
        industry_index: Optional[pd.DataFrame] = None,
        industry_col: str = "industry_l1",
    ) -> pd.DataFrame:
        """
        预测股票排序

        Args:
            df: 股票数据DataFrame
            daily_basic: 每日基本面数据
            fina_indicator: 财务指标数据
            hk_hold: 港资持股数据
            moneyflow: 资金流向数据
            industry_index: 行业指数数据
            industry_col: 行业分类字段名

        Returns:
            包含预测分数的DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        # 准备特征
        df_features = self.prepare_features(
            df,
            daily_basic=daily_basic,
            fina_indicator=fina_indicator,
            hk_hold=hk_hold,
            moneyflow=moneyflow,
            industry_index=industry_index,
            industry_col=industry_col,
        )

        # 获取特征名称
        feature_cols = self.feature_names

        # 检查缺失特征
        missing_features = [f for f in feature_cols if f not in df_features.columns]
        if missing_features:
            logger.warning(f"缺失特征列（将填充0）: {missing_features[:5]}...")
            for f in missing_features:
                df_features[f] = 0.0

        # 预测
        scores = self.model.predict(df_features[feature_cols].values)

        # 添加预测分数
        df_result = df_features.copy()
        df_result["rank_score"] = scores
        df_result["rank"] = df_result["rank_score"].rank(ascending=False)

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
        df_importance = pd.DataFrame({"feature": self.feature_names, "importance": importance}).sort_values(
            "importance", ascending=False
        )

        return df_importance


if __name__ == "__main__":
    # 测试排序模型
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")

    # 模拟多只股票的数据
    stock_codes = ["sh.600000", "sh.600004", "sh.600006"]
    all_data = []

    for code in stock_codes:
        close = 10 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        turnover = np.random.uniform(0.01, 0.1, len(dates))

        for i, date in enumerate(dates):
            all_data.append({"date": date, "code": code, "close": close[i], "turnover": turnover[i]})

    df = pd.DataFrame(all_data)

    # 测试模型
    print("测试排序模型...")
    config = {"lgbm_params": {"num_leaves": 16, "max_depth": 4}}

    model = RankModelLGBM(config)

    # 创建标签
    labels = model.create_ranking_label(df)
    df = df.dropna()
    labels = labels.loc[df.index]

    # 创建分组信息
    group_info = df.groupby("date").size()

    # 训练模型
    model.train(df, labels, group_info)

    # 预测
    df_predict = model.predict(df.tail(30))

    print("\n预测结果（最后30条）:")
    print(df_predict[["date", "code", "rank_score", "rank"]].head(10))

    # 特征重要性
    importance = model.get_feature_importance()
    print("\n特征重要性（Top 10）:")
    print(importance.head(10))
