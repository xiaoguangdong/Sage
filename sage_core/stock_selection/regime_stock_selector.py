"""
分 Regime 选股模型 — 按市场状态训练独立专家模型

核心思想：
- 牛市/震荡/熊市的因子规律不同，一个模型打天下必然过拟合
- 按趋势模型输出的 RISK_ON/NEUTRAL/RISK_OFF 分别训练三个 StockSelector
- 推理时根据当天 regime 选择对应模型打分
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector
from sage_core.trend.trend_model import TrendModelConfig, TrendModelRuleV2

logger = logging.getLogger(__name__)

# Regime 常量（与趋势模型状态对齐）
RISK_OFF = 0
NEUTRAL = 1
RISK_ON = 2

REGIME_NAMES = {RISK_OFF: "bear", NEUTRAL: "neutral", RISK_ON: "bull"}


@dataclass
class RegimeSelectionConfig:
    """分 Regime 选股配置"""

    base_config: SelectionConfig = field(default_factory=SelectionConfig)

    # 各 regime 的 LightGBM 参数覆盖（key: regime int）
    regime_lgbm_overrides: Dict[int, Dict] = field(default_factory=lambda: {
        RISK_ON: {"min_data_in_leaf": 500, "lambda_l2": 10.0},
        NEUTRAL: {"min_data_in_leaf": 500, "lambda_l2": 10.0},
        RISK_OFF: {"min_data_in_leaf": 800, "lambda_l2": 15.0},
    })

    # 趋势模型配置
    trend_config: TrendModelConfig = field(default_factory=TrendModelConfig)

    # 最小样本量（低于此值回退到全量模型）
    min_samples_per_regime: int = 200

    # 硬分配 vs 软分配
    hard_assignment: bool = True  # True=只用对应regime数据，False=全量+软权重

    # 软分配权重（hard_assignment=False 时生效）
    soft_weight_same: float = 1.0
    soft_weight_adjacent: float = 0.3
    soft_weight_opposite: float = 0.05

    # 时间衰减（半衰期，交易日数；0=不衰减）
    time_decay_half_life: int = 500  # ~2年交易日


class RegimeStockSelector:
    """分 Regime 选股器（Mixture of Experts）"""

    def __init__(self, config: Optional[RegimeSelectionConfig] = None):
        self.config = config or RegimeSelectionConfig()
        self.models: Dict[int, StockSelector] = {}
        self.fallback_model: Optional[StockSelector] = None
        self.trend_model = TrendModelRuleV2(self.config.trend_config)
        self.is_trained = False
        self.train_stats: Dict[int, Dict] = {}

    def _make_sub_config(self, regime: int) -> SelectionConfig:
        """为指定 regime 创建子模型配置"""
        cfg = copy.deepcopy(self.config.base_config)
        overrides = self.config.regime_lgbm_overrides.get(regime, {})
        if overrides:
            cfg.lgbm_params = {**cfg.lgbm_params, **overrides}
        return cfg

    def generate_regime_labels(
        self, df_index: pd.DataFrame, dates: pd.Series
    ) -> pd.Series:
        """
        用趋势模型为每个交易日生成 regime 标签

        Args:
            df_index: 指数数据（需含 close 列，按日期排序）
            dates: 需要标签的日期序列

        Returns:
            与 dates 对齐的 regime 标签 Series
        """
        result = self.trend_model.predict(df_index, return_history=True)
        states = result.diagnostics["states"]

        # 构建日期→状态映射
        idx_dates = df_index["date"] if "date" in df_index.columns else df_index.index
        if hasattr(idx_dates, "values"):
            idx_dates = idx_dates.values
        date_to_state = dict(zip(pd.to_datetime(idx_dates), states))

        # 映射到目标日期
        regime_labels = pd.Series(
            [date_to_state.get(pd.Timestamp(d), NEUTRAL) for d in dates],
            index=dates.index,
            dtype=int,
        )
        return regime_labels

    def fit(
        self,
        df: pd.DataFrame,
        regime_labels: pd.Series,
    ) -> "RegimeStockSelector":
        """
        按 regime 分别训练选股模型

        Args:
            df: 股票数据（含特征列）
            regime_labels: 与 df 行对齐的 regime 标签（0/1/2）
        """
        date_col = self.config.base_config.date_col

        # 训练全量回退模型
        logger.info("训练全量回退模型...")
        self.fallback_model = StockSelector(copy.deepcopy(self.config.base_config))
        self.fallback_model.fit(df)

        # 按 regime 分别训练
        for regime in [RISK_OFF, NEUTRAL, RISK_ON]:
            name = REGIME_NAMES[regime]
            n_target = int((regime_labels == regime).sum())

            self.train_stats[regime] = {"n_samples": n_target}

            if n_target < self.config.min_samples_per_regime:
                logger.warning(
                    f"[{name}] 样本不足 ({n_target} < {self.config.min_samples_per_regime})，使用全量模型"
                )
                self.models[regime] = self.fallback_model
                continue

            sub_config = self._make_sub_config(regime)
            model = StockSelector(sub_config)

            if self.config.hard_assignment:
                # 硬分配：只用对应 regime 的数据
                mask = regime_labels == regime
                df_regime = df[mask].reset_index(drop=True)
                dates_regime = df_regime[date_col] if date_col in df_regime.columns else None
                weights = self._build_time_decay_weights(dates_regime)
                n_dates = df_regime[date_col].nunique() if date_col in df_regime.columns else 0
                logger.info(f"[{name}] 硬分配训练: {n_target} 样本, {n_dates} 交易日")
                model.fit(df_regime, sample_weight=weights)
            else:
                # 软分配：全量数据 + 软权重 × 时间衰减
                dates_series = df[date_col] if date_col in df.columns else None
                weights = self._build_soft_weights(regime_labels, regime, dates=dates_series)
                logger.info(f"[{name}] 软分配训练: {n_target} 目标样本, {len(df)} 总样本")
                model.fit(df, sample_weight=weights)

            self.models[regime] = model

            # 记录特征信息
            self.train_stats[regime]["n_features"] = (
                len(model.feature_cols) if model.feature_cols else 0
            )
            if hasattr(model, "feature_ic_stats"):
                top_features = sorted(
                    model.feature_ic_stats.items(),
                    key=lambda x: abs(x[1]["mean_ic"]),
                    reverse=True,
                )[:5]
                self.train_stats[regime]["top_features"] = [
                    (f, round(s["mean_ic"], 4)) for f, s in top_features
                ]

        self.is_trained = True
        self._log_summary()
        return self

    def _build_time_decay_weights(self, dates: Optional[pd.Series] = None) -> Optional[np.ndarray]:
        """纯时间衰减权重（硬分配用）"""
        cfg = self.config
        if cfg.time_decay_half_life <= 0 or dates is None:
            return None
        ts = pd.to_datetime(dates)
        days_ago = (ts.max() - ts).dt.days.values.astype(float)
        half_life_cal = cfg.time_decay_half_life * 365.0 / 252.0
        return np.power(0.5, days_ago / half_life_cal)

    def _build_soft_weights(
        self, regime_labels: pd.Series, target_regime: int, dates: Optional[pd.Series] = None
    ) -> np.ndarray:
        """构建软分配权重 × 时间衰减权重"""
        cfg = self.config
        weights = np.full(len(regime_labels), cfg.soft_weight_opposite)
        adjacent = {
            RISK_OFF: [NEUTRAL],
            NEUTRAL: [RISK_OFF, RISK_ON],
            RISK_ON: [NEUTRAL],
        }
        for adj in adjacent.get(target_regime, []):
            weights[regime_labels == adj] = cfg.soft_weight_adjacent
        weights[regime_labels == target_regime] = cfg.soft_weight_same

        # 时间衰减
        if cfg.time_decay_half_life > 0 and dates is not None:
            ts = pd.to_datetime(dates)
            max_date = ts.max()
            days_ago = (max_date - ts).dt.days.values.astype(float)
            decay = np.power(0.5, days_ago / (cfg.time_decay_half_life * 365.0 / 252.0))
            weights = weights * decay

        return weights

    def predict(self, df: pd.DataFrame, regime: int) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        model = self.models.get(regime, self.fallback_model)
        return model.predict(df)

    def predict_prepared(self, df_features: pd.DataFrame, regime: int) -> pd.DataFrame:
        """对已计算好特征的数据直接预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        model = self.models.get(regime, self.fallback_model)
        return model.predict_prepared(df_features)

    def predict_with_index(
        self, df: pd.DataFrame, df_index: pd.DataFrame, trade_date: str
    ) -> pd.DataFrame:
        """
        自动判断 regime 并预测（端到端接口）

        Args:
            df: 待预测股票数据
            df_index: 指数历史数据（用于趋势判断）
            trade_date: 当前交易日期
        """
        trend_result = self.trend_model.predict(df_index)
        regime = trend_result.state
        logger.info(
            f"[{trade_date}] regime={REGIME_NAMES[regime]}, "
            f"confidence={trend_result.confidence:.2f}"
        )
        return self.predict(df, regime)

    def _log_summary(self):
        """打印训练摘要"""
        logger.info("=" * 50)
        logger.info("分 Regime 选股模型训练完成")
        for regime, stats in self.train_stats.items():
            name = REGIME_NAMES[regime]
            n = stats["n_samples"]
            nf = stats.get("n_features", "N/A")
            logger.info(f"  [{name}] 样本={n}, 特征={nf}")
            if "top_features" in stats:
                for feat, ic in stats["top_features"]:
                    logger.info(f"    {feat}: IC={ic}")
        logger.info("=" * 50)
