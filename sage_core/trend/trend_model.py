"""
趋势状态模型 - 支持规则/概率平滑/HMM多种方法

核心设计思想：
1. Layer 0（证据层）：MA差、斜率、价格位置、波动率、广度等指标
2. Layer 1（强度层）：将证据合成为连续趋势强度（-1 ~ +1）
3. Layer 2（状态层）：通过回滞阈值+最小持续期+粘性平滑输出离散状态

状态定义：
- 0: RISK_OFF（熊市/下跌）
- 1: NEUTRAL（震荡/中性）
- 2: RISK_ON（牛市/上涨）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class TrendState:
    """趋势状态输出"""
    state: int                          # 0/1/2: RISK_OFF/NEUTRAL/RISK_ON
    state_name: str                     # 状态名称
    confidence: float                   # 置信度 [0, 1]
    trend_strength: float               # 趋势强度 [-1, 1]
    position_suggestion: float          # 建议仓位 [0, 1]
    prob_risk_on: float                 # RISK_ON概率
    prob_neutral: float                 # NEUTRAL概率
    prob_risk_off: float                # RISK_OFF概率
    reasons: List[str] = field(default_factory=list)  # 状态原因
    diagnostics: Dict = field(default_factory=dict)   # 诊断信息


@dataclass
class TrendModelConfig:
    """趋势模型配置"""
    # 均线参数
    ma_short: int = 20
    ma_medium: int = 60
    ma_long: int = 120

    # 入场确认天数（与原版均线确认一致）
    confirmation_periods: int = 5       # 连续满足条件N天才切换

    # 退出宽容期（核心改进：解决断续问题）
    exit_tolerance: int = 8             # 条件不满足后宽容N天才退出

    # 最小持续期
    min_hold_periods: int = 10          # 状态切换后强制保持N天

    # P0: 极端行情熔断
    circuit_breaker_1d: float = -0.035  # 单日跌幅阈值（强制RISK_OFF）
    circuit_breaker_3d: float = -0.06   # 连续3日累计跌幅阈值
    circuit_breaker_5d: float = -0.08   # 连续5日累计跌幅阈值
    circuit_breaker_drawdown: float = -0.08  # 从20日高点回撤阈值
    circuit_breaker_hold: int = 5       # 熔断后强制RISK_OFF天数

    # P1: 波动率自适应退出宽容期
    vol_adaptive: bool = True           # 是否启用波动率自适应
    vol_window: int = 20                # 当前波动率窗口
    vol_median_window: int = 120        # 中位波动率窗口

    # P2: 多时间框架确认
    mtf_enabled: bool = True            # 是否启用多时间框架
    ma_weekly: int = 13                 # 周线级别均线（≈60日/5）

    # 仓位映射
    position_risk_on: Tuple[float, float] = (0.70, 0.95)
    position_neutral: Tuple[float, float] = (0.30, 0.60)
    position_risk_off: Tuple[float, float] = (0.0, 0.25)


# ============================================================================
# 核心模型：趋势强度 + 回滞状态机
# ============================================================================

class TrendModelRuleV2:
    """
    趋势状态模型 V2 - 规则版本（改进版）

    核心逻辑（与原版均线确认一致）：
    - 牛市条件：MA_short > MA_medium AND close > MA_short AND slope > 0
    - 熊市条件：MA_short < MA_medium AND close < MA_short AND slope < 0

    改进点（解决断续问题）：
    1. 入场确认：连续N天满足条件才切换（保留原版逻辑）
    2. 退出宽容期：条件不满足后宽容N天才退出（新增，解决断续）
    3. 最小持续期：状态切换后强制保持N天（新增，防抖动）
    """

    def __init__(self, config: Optional[TrendModelConfig] = None):
        self.config = config or TrendModelConfig()

    def predict(self, df_index: pd.DataFrame, return_history: bool = False) -> TrendState:
        if 'close' not in df_index.columns:
            raise ValueError("DataFrame必须包含'close'列")

        cfg = self.config
        close = df_index['close']

        # 计算均线
        ma_short = close.rolling(cfg.ma_short, min_periods=cfg.ma_short).mean()
        ma_medium = close.rolling(cfg.ma_medium, min_periods=cfg.ma_medium).mean()
        slope = ma_short.diff()

        # P0: 熔断用日收益率
        daily_ret = close.pct_change()
        ret_3d = close.pct_change(3)
        ret_5d = close.pct_change(5)
        rolling_high = close.rolling(20, min_periods=1).max()
        drawdown = (close - rolling_high) / rolling_high

        # P1: 波动率自适应
        vol_current = daily_ret.rolling(cfg.vol_window, min_periods=cfg.vol_window).std()
        vol_median = vol_current.rolling(cfg.vol_median_window, min_periods=cfg.vol_median_window).median()

        # P2: 多时间框架（周线级别均线）
        ma_weekly = close.rolling(cfg.ma_weekly * 5, min_periods=cfg.ma_weekly * 5).mean()

        # 状态机遍历
        states = []
        trend_strength_list = []
        state = 1           # 当前状态
        confirm_count = 0   # 入场确认计数
        fail_count = 0      # 退出宽容计数
        hold_count = 0      # 最小持续期计数

        for i in range(len(df_index)):
            ms = ma_short.iloc[i]
            mm = ma_medium.iloc[i]
            sl = slope.iloc[i]
            cl = close.iloc[i]

            # 数据不足
            if pd.isna(ms) or pd.isna(mm) or pd.isna(sl):
                states.append(1)
                trend_strength_list.append(0.0)
                continue

            # 计算趋势强度（辅助信息，不参与状态判断）
            ts = self._calc_strength(cl, ms, mm, sl)
            trend_strength_list.append(ts)

            # P0: 极端行情熔断（优先级最高，覆盖一切）
            dr = daily_ret.iloc[i]
            r3 = ret_3d.iloc[i]
            r5 = ret_5d.iloc[i]
            dd = drawdown.iloc[i]
            _tripped = False
            for _val, _th in [(dr, cfg.circuit_breaker_1d), (r3, cfg.circuit_breaker_3d),
                              (r5, cfg.circuit_breaker_5d), (dd, cfg.circuit_breaker_drawdown)]:
                if not pd.isna(_val) and _val <= _th:
                    _tripped = True
                    break
            if _tripped:
                state = 0
                hold_count = cfg.circuit_breaker_hold
                fail_count = 0
                confirm_count = 0
                states.append(state)
                continue

            # 硬条件判断（与原版均线确认一致）
            bull = (ms > mm) and (cl > ms) and (sl > 0)
            bear = (ms < mm) and (cl < ms) and (sl < 0)

            # P1: 波动率自适应退出宽容期
            vc = vol_current.iloc[i]
            vm = vol_median.iloc[i]
            if cfg.vol_adaptive and not pd.isna(vc) and not pd.isna(vm) and vm > 0:
                vol_ratio = vm / max(vc, 1e-8)  # 波动率越高，ratio越小，宽容期越短
                adaptive_tolerance = max(3, int(cfg.exit_tolerance * np.clip(vol_ratio, 0.5, 1.5)))
            else:
                adaptive_tolerance = cfg.exit_tolerance

            # P2: 多时间框架确认
            mw = ma_weekly.iloc[i]
            if cfg.mtf_enabled and not pd.isna(mw):
                weekly_bull = cl > mw
                weekly_bear = cl < mw
            else:
                weekly_bull = True
                weekly_bear = True

            # 最小持续期：强制保持
            if hold_count > 0:
                hold_count -= 1
                states.append(state)
                continue

            # 状态机
            if state == 1:  # 震荡态
                if bull and weekly_bull:  # P2: 周线确认
                    confirm_count += 1
                    if confirm_count >= cfg.confirmation_periods:
                        state = 2
                        hold_count = cfg.min_hold_periods
                        confirm_count = 0
                        fail_count = 0
                elif bear and weekly_bear:  # P2: 周线确认
                    confirm_count += 1
                    if confirm_count >= cfg.confirmation_periods:
                        state = 0
                        hold_count = cfg.min_hold_periods
                        confirm_count = 0
                        fail_count = 0
                else:
                    confirm_count = 0

            elif state == 2:  # 牛市态
                if bull:
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count >= adaptive_tolerance:  # P1: 自适应宽容期
                        state = 1
                        fail_count = 0
                        confirm_count = 0

            elif state == 0:  # 熊市态
                if bear:
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count >= adaptive_tolerance:  # P1: 自适应宽容期
                        state = 1
                        fail_count = 0
                        confirm_count = 0

            states.append(state)

        # 构建输出
        final_state = states[-1]
        final_ts = trend_strength_list[-1]
        p_on, p_neu, p_off = self._calc_probability(final_ts)
        confidence = self._calc_confidence(final_state, p_on, p_off)
        position = self._calc_position(final_state, confidence)
        reasons = self._gen_reasons(
            final_state, ma_short.iloc[-1], ma_medium.iloc[-1], close.iloc[-1], slope.iloc[-1]
        )

        state_names = {0: 'RISK_OFF', 1: 'NEUTRAL', 2: 'RISK_ON'}
        result = TrendState(
            state=final_state,
            state_name=state_names[final_state],
            confidence=round(confidence, 4),
            trend_strength=round(final_ts, 4),
            position_suggestion=round(position, 4),
            prob_risk_on=round(p_on, 4),
            prob_neutral=round(p_neu, 4),
            prob_risk_off=round(p_off, 4),
            reasons=reasons,
            diagnostics={
                'ma_short': round(ma_short.iloc[-1], 2),
                'ma_medium': round(ma_medium.iloc[-1], 2),
                'current_price': round(close.iloc[-1], 2),
                'slope': round(slope.iloc[-1], 4),
            }
        )

        if return_history:
            result.diagnostics['states'] = states
            result.diagnostics['trend_strength'] = trend_strength_list

        logger.info(f"趋势状态: {result.state_name} (confidence={result.confidence:.2f})")
        return result

    # ---- 辅助方法 ----

    @staticmethod
    def _calc_strength(close: float, ma_s: float, ma_m: float, slope: float) -> float:
        """趋势强度（辅助指标，不参与状态判断）"""
        ma_diff = (ma_s - ma_m) / ma_m
        price_pos = (close - ma_s) / ma_s
        slope_norm = np.tanh(slope / ma_s * 100)
        return float(np.clip(0.5 * np.tanh(ma_diff * 15) + 0.3 * slope_norm + 0.2 * np.tanh(price_pos * 10), -1, 1))

    @staticmethod
    def _calc_probability(ts: float) -> Tuple[float, float, float]:
        k = 5.0
        p_on = 1 / (1 + np.exp(-k * (ts - 0.3)))
        p_off = 1 / (1 + np.exp(-k * (-ts - 0.3)))
        p_neu = max(0.0, 1 - p_on - p_off)
        total = p_on + p_neu + p_off
        return p_on / total, p_neu / total, p_off / total

    @staticmethod
    def _calc_confidence(state: int, p_on: float, p_off: float) -> float:
        if state == 2:
            return min(1.0, 0.5 + p_on * 0.5)
        elif state == 0:
            return min(1.0, 0.5 + p_off * 0.5)
        return 0.6

    def _calc_position(self, state: int, confidence: float) -> float:
        cfg = self.config
        if state == 2:
            lo, hi = cfg.position_risk_on
        elif state == 0:
            lo, hi = cfg.position_risk_off
        else:
            lo, hi = cfg.position_neutral
        return lo + (hi - lo) * confidence

    @staticmethod
    def _gen_reasons(state: int, ma_s: float, ma_m: float, close: float, slope: float) -> List[str]:
        reasons = []
        if ma_s > ma_m:
            reasons.append("MA短期>中期")
        else:
            reasons.append("MA短期<中期")
        if close > ma_s:
            reasons.append("价格>短期均线")
        else:
            reasons.append("价格<短期均线")
        if slope > 0:
            reasons.append("均线斜率向上")
        else:
            reasons.append("均线斜率向下")
        return reasons


# ============================================================================
# 旧版兼容：TrendModelRule（保持向后兼容）
# ============================================================================

class TrendModelRule(TrendModelRuleV2):
    """
    趋势状态模型（规则版本）- 兼容旧接口

    继承自 TrendModelRuleV2，保持向后兼容
    """

    def __init__(
        self,
        ma_short: int = 20,
        ma_medium: int = 60,
        ma_long: int = 120,
        **kwargs,
    ):
        config = TrendModelConfig(
            ma_short=ma_short,
            ma_medium=ma_medium,
            ma_long=ma_long,
        )
        super().__init__(config)


# ============================================================================
# HMM 版本（粘性HMM）
# ============================================================================

class TrendModelHMM:
    """
    趋势状态模型（HMM版本）- 粘性HMM

    特点：
    1. 使用趋势证据作为观测特征（而非原始收益）
    2. 转移矩阵增加粘性（留在原状态的概率更高）
    3. 观测分布使用t分布（更适合金融厚尾）
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.is_trained = False

        # 默认参数
        self.n_components = self.config.get('n_components', 3)
        self.sticky_factor = self.config.get('sticky_factor', 0.90)
        self.min_duration = self.config.get('min_duration', 5)

    def _compute_observations(self, df: pd.DataFrame) -> np.ndarray:
        """
        计算HMM观测特征（趋势证据）
        """
        close = df['close']

        # 趋势强度
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        trend_strength = (ma20 - ma60) / ma60

        # 波动率
        vol = close.pct_change().rolling(20).std() * np.sqrt(252)

        # 价格动量
        momentum = close.pct_change(20)

        # 组合观测
        obs = pd.DataFrame({
            'trend_strength': trend_strength,
            'volatility': vol,
            'momentum': momentum,
        }).dropna()

        return obs.values

    def train(self, df: pd.DataFrame) -> 'TrendModelHMM':
        """
        训练HMM模型
        """
        try:
            from hmmlearn import hmm
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("未安装hmmlearn，无法训练HMM模型")
            return self

        obs = self._compute_observations(df)

        if len(obs) < 100:
            logger.warning("数据量不足，无法训练HMM模型")
            return self

        # 标准化
        self.scaler = StandardScaler()
        obs_scaled = self.scaler.fit_transform(obs)

        # 训练HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type='full',
            n_iter=500,
            random_state=42,
        )

        # 初始化粘性转移矩阵
        transmat = np.full(
            (self.n_components, self.n_components),
            (1 - self.sticky_factor) / (self.n_components - 1)
        )
        np.fill_diagonal(transmat, self.sticky_factor)
        self.model.startprob_ = np.ones(self.n_components) / self.n_components
        self.model.transmat_ = transmat

        self.model.fit(obs_scaled)
        self.is_trained = True

        logger.info("HMM模型训练完成")
        return self

    def predict(self, df: pd.DataFrame) -> TrendState:
        """
        预测趋势状态
        """
        if not self.is_trained:
            logger.warning("HMM模型未训练，使用默认状态")
            return TrendState(
                state=1,
                state_name='NEUTRAL',
                confidence=0.5,
                trend_strength=0.0,
                position_suggestion=0.4,
                prob_risk_on=0.2,
                prob_neutral=0.6,
                prob_risk_off=0.2,
            )

        obs = self._compute_observations(df)
        obs_scaled = self.scaler.transform(obs)

        # 预测状态序列
        states = self.model.predict(obs_scaled)
        final_state = states[-1]

        # 计算概率
        probs = self.model.predict_proba(obs_scaled)[-1]

        # 状态映射（需要根据训练结果确定）
        # 这里简化处理，假设状态0=熊市，1=震荡，2=牛市
        state_names = {0: 'RISK_OFF', 1: 'NEUTRAL', 2: 'RISK_ON'}

        return TrendState(
            state=final_state,
            state_name=state_names.get(final_state, 'NEUTRAL'),
            confidence=round(max(probs), 4),
            trend_strength=0.0,  # HMM不直接输出强度
            position_suggestion=round(0.3 + max(probs) * 0.5, 4),
            prob_risk_on=round(probs[2] if len(probs) > 2 else 0.2, 4),
            prob_neutral=round(probs[1] if len(probs) > 1 else 0.6, 4),
            prob_risk_off=round(probs[0], 4),
        )


# ============================================================================
# LGBM 版本（待实现）
# ============================================================================

class TrendModelLGBM:
    """
    趋势状态模型（LightGBM版本）- 待实现

    计划特点：
    1. 使用多个趋势证据作为特征
    2. 目标变量：未来N日收益的风险调整后表现
    3. 输出：三分类概率
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.model = None
        self.is_trained = False
        logger.warning("LightGBM版本的趋势模型尚未实现，请使用TrendModelRuleV2")

    def train(self, df: pd.DataFrame, labels: pd.Series) -> 'TrendModelLGBM':
        """训练模型"""
        logger.warning("LightGBM版本暂未实现")
        return self

    def predict(self, df: pd.DataFrame) -> TrendState:
        """预测趋势状态"""
        return TrendState(
            state=1,
            state_name='NEUTRAL',
            confidence=0.5,
            trend_strength=0.0,
            position_suggestion=0.4,
            prob_risk_on=0.2,
            prob_neutral=0.6,
            prob_risk_off=0.2,
        )


# ============================================================================
# 工厂函数
# ============================================================================

def create_trend_model(
    model_type: str = 'rule',
    config: Optional[Dict] = None
) -> TrendModelRuleV2 | TrendModelHMM | TrendModelLGBM:
    """
    创建趋势模型工厂函数

    Args:
        model_type: 模型类型
            - 'rule' 或 'rule_v2': 改进版规则模型（推荐）
            - 'hmm': HMM模型
            - 'lgbm': LightGBM模型（未实现）
        config: 配置字典

    Returns:
        趋势模型实例
    """
    config = config or {}

    if model_type in ('rule', 'rule_v2'):
        model_config = TrendModelConfig()
        if 'rule_params' in config:
            params = config['rule_params']
            model_config = TrendModelConfig(
                ma_short=params.get('ma_short', 20),
                ma_medium=params.get('ma_medium', 60),
                ma_long=params.get('ma_long', 120),
                confirmation_periods=params.get('confirmation_periods', 3),
                exit_tolerance=params.get('exit_tolerance', 5),
                min_hold_periods=params.get('min_hold_periods', 7),
            )
        return TrendModelRuleV2(model_config)

    elif model_type == 'hmm':
        return TrendModelHMM(config.get('hmm_params', {}))

    elif model_type == 'lgbm':
        return TrendModelLGBM(config.get('lgbm_params', {}))

    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

    # 模拟指数数据（带趋势+噪声）
    trend = np.concatenate([
        np.linspace(3000, 4000, 500),   # 上涨
        np.linspace(4000, 3500, 300),   # 回调
        np.linspace(3500, 3800, 400),   # 震荡上行
        np.linspace(3800, 3200, 300),   # 下跌
        np.linspace(3200, 3400, 200),   # 反弹
    ])
    noise = np.random.randn(len(dates)) * 20
    close = trend[:len(dates)] + noise

    df = pd.DataFrame({
        'date': dates,
        'close': close,
        'high': close + np.abs(np.random.randn(len(dates))) * 10,
        'low': close - np.abs(np.random.randn(len(dates))) * 10,
    })

    # 测试规则模型
    print("=" * 60)
    print("测试趋势模型 V2")
    print("=" * 60)

    model = create_trend_model('rule')
    result = model.predict(df, return_history=True)

    print(f"\n预测结果:")
    print(f"  状态: {result.state_name} ({result.state})")
    print(f"  置信度: {result.confidence:.2%}")
    print(f"  趋势强度: {result.trend_strength:.2f}")
    print(f"  建议仓位: {result.position_suggestion:.2%}")
    print(f"  牛市概率: {result.prob_risk_on:.2%}")
    print(f"  震荡概率: {result.prob_neutral:.2%}")
    print(f"  熊市概率: {result.prob_risk_off:.2%}")
    print(f"  原因: {result.reasons}")
    print(f"\n  诊断信息: {result.diagnostics}")
