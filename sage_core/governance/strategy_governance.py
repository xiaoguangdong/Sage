from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from sage_core.stock_selection.stock_selector import SelectionConfig, StockSelector
from sage_core.stock_selection.multi_alpha_selector import MultiAlphaStockSelector

logger = logging.getLogger(__name__)


SIGNAL_SCHEMA = [
    "trade_date",
    "ts_code",
    "score",
    "rank",
    "confidence",
    "model_version",
]

SUPPORTED_STRATEGIES = (
    "seed_balance_strategy",
    "balance_strategy_v1",
    "positive_strategy_v1",
    "value_strategy_v1",
    "satellite_strategy_v1",
)

STRATEGY_ALIASES = {
    "seed_banlance_strategy": "seed_balance_strategy",
}


def normalize_strategy_id(strategy_id: str) -> str:
    key = (strategy_id or "").strip()
    normalized = STRATEGY_ALIASES.get(key, key)
    if normalized not in SUPPORTED_STRATEGIES:
        raise ValueError(f"不支持的策略ID: {strategy_id}")
    return normalized


def _validate_signal_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in SIGNAL_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"信号缺少必要字段: {missing}")
    return df[SIGNAL_SCHEMA].copy()


def _format_signal_frame(
    raw: pd.DataFrame,
    score_col: str,
    ts_code_col: str,
    trade_date: str,
    model_version: str,
    top_n: int,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=SIGNAL_SCHEMA)

    data = raw[[ts_code_col, score_col]].copy()
    data = data.rename(columns={ts_code_col: "ts_code", score_col: "score"})
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data = data.dropna(subset=["score"])
    data = data.sort_values("score", ascending=False).reset_index(drop=True)

    total = int(len(data))
    if total == 0:
        return pd.DataFrame(columns=SIGNAL_SCHEMA)

    if top_n > 0:
        data = data.head(top_n).copy()
    data["rank"] = np.arange(1, len(data) + 1, dtype=int)

    if total > 1:
        data["confidence"] = 1.0 - (data["rank"] - 1) / (total - 1)
    else:
        data["confidence"] = 1.0

    data["confidence"] = data["confidence"].clip(lower=0.0, upper=1.0)
    data["trade_date"] = str(trade_date)
    data["model_version"] = model_version
    return _validate_signal_schema(data)


def _normalize_trade_date(value: str | int | pd.Timestamp) -> str:
    try:
        ts = pd.to_datetime(value)
        return ts.strftime("%Y%m%d")
    except Exception:
        return str(value)


def decide_auto_promotion(
    evaluation_df: pd.DataFrame,
    current_champion: str,
    challengers: Tuple[str, ...],
    as_of_date: Optional[str] = None,
    enabled: bool = False,
    manual_mode: bool = False,
    allow_when_manual: bool = False,
    consecutive_periods: int = 3,
    max_drawdown_tolerance: float = 0.02,
    max_turnover_multiplier: float = 1.2,
    min_cost_return_diff: float = 0.0,
    min_sharpe_diff: float = 0.0,
    min_data_quality: float = 0.95,
) -> Dict[str, object]:
    """
    自动晋升决策：
    - 仅当 enabled=True 时生效
    - Challenger 需连续N个评估周期满足硬门槛
    - 多候选时按综合得分差（challenger - champion）最大者晋升
    """
    current_champion = normalize_strategy_id(current_champion)
    challengers = tuple(normalize_strategy_id(s) for s in challengers)

    decision = {
        "enabled": bool(enabled),
        "current_champion": current_champion,
        "next_champion": current_champion,
        "promoted": False,
        "reason": "disabled",
        "candidate": None,
        "score_diff": 0.0,
        "periods_checked": 0,
    }
    if not enabled:
        return decision
    if manual_mode and not allow_when_manual:
        decision["reason"] = "manual_mode"
        return decision
    if evaluation_df is None or evaluation_df.empty:
        decision["reason"] = "no_evaluation_data"
        return decision

    required_cols = {"strategy_id", "trade_date", "cost_return", "max_drawdown", "sharpe", "turnover"}
    if not required_cols.issubset(evaluation_df.columns):
        decision["reason"] = "invalid_evaluation_schema"
        return decision

    data = evaluation_df.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
    data = data.dropna(subset=["trade_date"])
    if as_of_date is not None:
        cutoff = pd.to_datetime(_normalize_trade_date(as_of_date), format="%Y%m%d", errors="coerce")
        if pd.notna(cutoff):
            data = data[data["trade_date"] <= cutoff]
    if data.empty:
        decision["reason"] = "no_data_before_cutoff"
        return decision

    if "data_quality" not in data.columns:
        data["data_quality"] = 1.0

    champion_rows = data[data["strategy_id"] == current_champion].copy()
    if champion_rows.empty:
        decision["reason"] = "champion_metrics_missing"
        return decision

    best_candidate = None
    best_diff = -np.inf
    best_periods = 0

    for challenger in challengers:
        if challenger == current_champion:
            continue
        challenger_rows = data[data["strategy_id"] == challenger].copy()
        if challenger_rows.empty:
            continue

        merged = challenger_rows.merge(
            champion_rows,
            on="trade_date",
            how="inner",
            suffixes=("_c", "_ch"),
        ).sort_values("trade_date")
        recent = merged.tail(consecutive_periods)
        if len(recent) < consecutive_periods:
            continue

        gate = (
            (recent["cost_return_c"] >= recent["cost_return_ch"] + min_cost_return_diff)
            & (recent["sharpe_c"] >= recent["sharpe_ch"] + min_sharpe_diff)
            & (recent["turnover_c"] <= recent["turnover_ch"] * max_turnover_multiplier)
            & (recent["data_quality_c"] >= min_data_quality)
            & (recent["max_drawdown_c"].abs() <= recent["max_drawdown_ch"].abs() + max_drawdown_tolerance)
        )
        if not bool(gate.all()):
            continue

        challenger_score = (
            recent["cost_return_c"] * 0.4
            + recent["sharpe_c"] * 0.3
            - recent["max_drawdown_c"].abs() * 0.2
            - recent["turnover_c"] * 0.1
        )
        champion_score = (
            recent["cost_return_ch"] * 0.4
            + recent["sharpe_ch"] * 0.3
            - recent["max_drawdown_ch"].abs() * 0.2
            - recent["turnover_ch"] * 0.1
        )
        score_diff = float((challenger_score - champion_score).mean())
        if score_diff > best_diff:
            best_diff = score_diff
            best_candidate = challenger
            best_periods = len(recent)

    if best_candidate and best_diff > 0:
        decision.update({
            "next_champion": best_candidate,
            "promoted": True,
            "reason": "challenger_outperformed",
            "candidate": best_candidate,
            "score_diff": round(best_diff, 6),
            "periods_checked": int(best_periods),
        })
    else:
        decision["reason"] = "no_eligible_challenger"
    return decision


@dataclass
class StrategyGovernanceConfig:
    active_champion_id: str = "seed_balance_strategy"
    champion_source: str = "manual"  # manual / auto
    manual_effective_date: Optional[str] = None
    manual_reason: Optional[str] = None
    challengers: Tuple[str, ...] = (
        "balance_strategy_v1",
        "positive_strategy_v1",
        "value_strategy_v1",
        "satellite_strategy_v1",
    )

    def normalized_active_champion_id(self) -> str:
        return normalize_strategy_id(self.active_champion_id)

    def normalized_challengers(self) -> Tuple[str, ...]:
        return tuple(normalize_strategy_id(s) for s in self.challengers)


@dataclass
class ChampionSwitchAudit:
    """Champion 切换审计记录"""

    timestamp: str  # ISO格式时间戳
    previous_champion: str
    new_champion: str
    switch_type: str  # manual / auto_promotion / auto_demotion
    effective_date: str
    reason: str
    operator: str = "system"  # 操作人/触发源
    evaluation_metrics: Optional[Dict] = None  # 切换时的评估指标快照
    approved_by: Optional[str] = None  # 审批人（可选）

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "previous_champion": self.previous_champion,
            "new_champion": self.new_champion,
            "switch_type": self.switch_type,
            "effective_date": self.effective_date,
            "reason": self.reason,
            "operator": self.operator,
            "evaluation_metrics": self.evaluation_metrics,
            "approved_by": self.approved_by,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChampionSwitchAudit":
        return cls(
            timestamp=data["timestamp"],
            previous_champion=data["previous_champion"],
            new_champion=data["new_champion"],
            switch_type=data["switch_type"],
            effective_date=data["effective_date"],
            reason=data["reason"],
            operator=data.get("operator", "system"),
            evaluation_metrics=data.get("evaluation_metrics"),
            approved_by=data.get("approved_by"),
        )


def switch_champion_manually(
    config: StrategyGovernanceConfig,
    new_champion: str,
    effective_date: str,
    reason: str,
    operator: str = "manual",
    evaluation_metrics: Optional[Dict] = None,
    approved_by: Optional[str] = None,
) -> Tuple[StrategyGovernanceConfig, ChampionSwitchAudit]:
    """
    手动切换 Champion

    Args:
        config: 当前治理配置
        new_champion: 新 Champion 策略ID
        effective_date: 生效日期 (YYYYMMDD)
        reason: 切换原因
        operator: 操作人
        evaluation_metrics: 评估指标快照
        approved_by: 审批人

    Returns:
        (更新后的配置, 审计记录)
    """
    from datetime import datetime

    new_champion = normalize_strategy_id(new_champion)
    previous_champion = config.normalized_active_champion_id()

    if new_champion == previous_champion:
        logger.warning(f"Champion 未变化: {new_champion}")
        return config, ChampionSwitchAudit(
            timestamp=datetime.now().isoformat(),
            previous_champion=previous_champion,
            new_champion=new_champion,
            switch_type="manual_no_change",
            effective_date=effective_date,
            reason=reason,
            operator=operator,
        )

    # 创建审计记录
    audit = ChampionSwitchAudit(
        timestamp=datetime.now().isoformat(),
        previous_champion=previous_champion,
        new_champion=new_champion,
        switch_type="manual",
        effective_date=effective_date,
        reason=reason,
        operator=operator,
        evaluation_metrics=evaluation_metrics,
        approved_by=approved_by,
    )

    # 更新配置
    updated_config = replace(
        config,
        active_champion_id=new_champion,
        champion_source="manual",
        manual_effective_date=effective_date,
        manual_reason=reason,
    )

    logger.info(
        f"Champion 切换: {previous_champion} -> {new_champion}, "
        f"生效日期: {effective_date}, 原因: {reason}"
    )

    return updated_config, audit


def save_audit_log(
    audit: ChampionSwitchAudit,
    audit_dir: Path,
) -> Path:
    """
    保存审计日志到文件

    Args:
        audit: 审计记录
        audit_dir: 审计日志目录

    Returns:
        审计日志文件路径
    """
    import json

    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    # 文件名: champion_switch_YYYYMMDD_HHMMSS.json
    from datetime import datetime
    ts = datetime.fromisoformat(audit.timestamp)
    filename = f"champion_switch_{ts.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = audit_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(audit.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info(f"审计日志已保存: {filepath}")
    return filepath


def load_audit_history(
    audit_dir: Path,
    limit: int = 100,
) -> pd.DataFrame:
    """
    加载审计历史记录

    Args:
        audit_dir: 审计日志目录
        limit: 最大加载条数

    Returns:
        审计历史 DataFrame
    """
    import json
    import glob

    audit_dir = Path(audit_dir)
    if not audit_dir.exists():
        return pd.DataFrame()

    files = sorted(glob.glob(str(audit_dir / "champion_switch_*.json")), reverse=True)[:limit]

    records = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                records.append(json.load(fp))
        except Exception as e:
            logger.warning(f"无法加载审计日志 {f}: {e}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp", ascending=False)


# ==================== 自动降级机制 ====================

# 保守基线策略（降级回退目标）
BASELINE_CONSERVATIVE_ID = "baseline_conservative"

# 降级阈值配置
DEFAULT_DOWNGRADE_THRESHOLDS = {
    "max_drawdown_limit": -0.15,  # 最大回撤阈值（超过则触发降级）
    "min_sharpe": 0.0,  # 最小夏普比率
    "max_turnover": 2.0,  # 最大换手率倍数（相对于基准）
    "consecutive_loss_periods": 3,  # 连续亏损周期数
    "min_cost_return": -0.05,  # 最小成本后收益
}


class BaselineConservativeStrategy:
    """保守基线策略（用于降级回退）"""

    def __init__(
        self,
        strategy_id: str = BASELINE_CONSERVATIVE_ID,
        version: str = "v1.0.0",
    ):
        self.strategy_id = strategy_id
        self.model_version = f"{strategy_id}@{version}"
        # 保守策略：只选择大盘蓝筹，低波动，高股息
        self.selection_criteria = {
            "min_market_cap": 5e10,  # 最小市值 500亿
            "max_volatility": 0.02,  # 最大日波动率 2%
            "min_dividend_yield": 0.02,  # 最小股息率 2%
            "top_n": 10,
        }

    def generate_signals(
        self,
        data: pd.DataFrame,
        trade_date: str,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        生成保守策略信号

        Args:
            data: 股票数据（需包含 total_mv, volatility, dividend_yield 等字段）
            trade_date: 交易日期
            top_n: 选股数量

        Returns:
            信号 DataFrame
        """
        if data is None or data.empty:
            return pd.DataFrame(columns=SIGNAL_SCHEMA)

        # 标准化日期
        if 'trade_date' in data.columns:
            data = data.rename(columns={'trade_date': 'date'})

        # 筛选日期
        date_norm = _normalize_trade_date(trade_date)
        if 'date' in data.columns:
            data = data[data['date'].astype(str) == date_norm]

        # 保守筛选
        filtered = data.copy()

        # 市值筛选
        if 'total_mv' in filtered.columns:
            filtered = filtered[filtered['total_mv'] >= self.selection_criteria['min_market_cap']]

        # 波动率筛选（如果有）
        if 'volatility_20d' in filtered.columns:
            filtered = filtered[filtered['volatility_20d'] <= self.selection_criteria['max_volatility']]

        # 股息率筛选（如果有）
        if 'dv_ttm' in filtered.columns:
            filtered = filtered[filtered['dv_ttm'] >= self.selection_criteria['min_dividend_yield']]

        # 计算保守评分
        score = pd.Series(0.0, index=filtered.index)

        # 市值越大越好（流动性）
        if 'total_mv' in filtered.columns:
            score += (filtered['total_mv'] / filtered['total_mv'].max()).fillna(0) * 0.4

        # 波动率越低越好
        if 'volatility_20d' in filtered.columns:
            vol_score = 1 - (filtered['volatility_20d'] / filtered['volatility_20d'].max()).fillna(0)
            score += vol_score * 0.3

        # 股息率越高越好
        if 'dv_ttm' in filtered.columns:
            div_score = (filtered['dv_ttm'] / filtered['dv_ttm'].max()).fillna(0)
            score += div_score * 0.3

        # 排序选择
        filtered = filtered.copy()
        filtered['score'] = score
        filtered = filtered.sort_values('score', ascending=False).head(top_n)

        return _format_signal_frame(
            raw=filtered,
            score_col='score',
            ts_code_col='ts_code' if 'ts_code' in filtered.columns else 'code',
            trade_date=date_norm,
            model_version=self.model_version,
            top_n=top_n,
        )


def check_downgrade_conditions(
    evaluation_df: pd.DataFrame,
    current_champion: str,
    thresholds: Optional[Dict] = None,
    lookback_periods: int = 3,
) -> Dict[str, object]:
    """
    检查是否需要触发自动降级

    Args:
        evaluation_df: 评估指标 DataFrame
        current_champion: 当前 Champion
        thresholds: 降级阈值配置
        lookback_periods: 回看周期数

    Returns:
        降级决策结果
    """
    from datetime import datetime

    thresholds = thresholds or DEFAULT_DOWNGRADE_THRESHOLDS
    current_champion = normalize_strategy_id(current_champion)

    result = {
        "should_downgrade": False,
        "reason": "none",
        "triggered_thresholds": [],
        "metrics_snapshot": None,
        "current_champion": current_champion,
        "fallback_strategy": BASELINE_CONSERVATIVE_ID,
    }

    if evaluation_df is None or evaluation_df.empty:
        result["reason"] = "no_evaluation_data"
        return result

    # 获取 Champion 最近评估数据
    champion_data = evaluation_df[evaluation_df["strategy_id"] == current_champion].copy()
    if champion_data.empty:
        result["reason"] = "champion_metrics_missing"
        return result

    champion_data = champion_data.sort_values("trade_date", ascending=False).head(lookback_periods)
    if len(champion_data) < lookback_periods:
        result["reason"] = "insufficient_data"
        return result

    # 检查各项阈值
    triggered = []

    # 1. 最大回撤检查
    if "max_drawdown" in champion_data.columns:
        max_dd = champion_data["max_drawdown"].min()
        if max_dd < thresholds["max_drawdown_limit"]:
            triggered.append({
                "metric": "max_drawdown",
                "value": float(max_dd),
                "threshold": thresholds["max_drawdown_limit"],
            })

    # 2. 夏普比率检查
    if "sharpe" in champion_data.columns:
        min_sharpe = champion_data["sharpe"].min()
        if min_sharpe < thresholds["min_sharpe"]:
            triggered.append({
                "metric": "sharpe",
                "value": float(min_sharpe),
                "threshold": thresholds["min_sharpe"],
            })

    # 3. 成本后收益检查
    if "cost_return" in champion_data.columns:
        min_cost_return = champion_data["cost_return"].min()
        if min_cost_return < thresholds["min_cost_return"]:
            triggered.append({
                "metric": "cost_return",
                "value": float(min_cost_return),
                "threshold": thresholds["min_cost_return"],
            })

    # 4. 连续亏损检查
    if "cost_return" in champion_data.columns:
        consecutive_loss = (champion_data["cost_return"] < 0).sum()
        if consecutive_loss >= thresholds["consecutive_loss_periods"]:
            triggered.append({
                "metric": "consecutive_loss",
                "value": int(consecutive_loss),
                "threshold": thresholds["consecutive_loss_periods"],
            })

    # 判断是否降级
    if triggered:
        result["should_downgrade"] = True
        result["reason"] = "thresholds_breached"
        result["triggered_thresholds"] = triggered
        result["metrics_snapshot"] = {
            "max_drawdown": float(champion_data["max_drawdown"].min()) if "max_drawdown" in champion_data.columns else None,
            "sharpe": float(champion_data["sharpe"].mean()) if "sharpe" in champion_data.columns else None,
            "cost_return": float(champion_data["cost_return"].mean()) if "cost_return" in champion_data.columns else None,
            "turnover": float(champion_data["turnover"].mean()) if "turnover" in champion_data.columns else None,
        }

    return result


def execute_auto_downgrade(
    config: StrategyGovernanceConfig,
    evaluation_df: pd.DataFrame,
    thresholds: Optional[Dict] = None,
    lookback_periods: int = 3,
    effective_date: Optional[str] = None,
) -> Tuple[StrategyGovernanceConfig, Optional[ChampionSwitchAudit]]:
    """
    执行自动降级

    Args:
        config: 当前治理配置
        evaluation_df: 评估指标 DataFrame
        thresholds: 降级阈值配置
        lookback_periods: 回看周期数
        effective_date: 生效日期

    Returns:
        (更新后的配置, 审计记录或 None)
    """
    from datetime import datetime

    # 检查降级条件
    downgrade_check = check_downgrade_conditions(
        evaluation_df=evaluation_df,
        current_champion=config.active_champion_id,
        thresholds=thresholds,
        lookback_periods=lookback_periods,
    )

    if not downgrade_check["should_downgrade"]:
        return config, None

    # 执行降级
    previous_champion = config.normalized_active_champion_id()
    new_champion = BASELINE_CONSERVATIVE_ID
    effective_date = effective_date or datetime.now().strftime("%Y%m%d")

    # 创建审计记录
    audit = ChampionSwitchAudit(
        timestamp=datetime.now().isoformat(),
        previous_champion=previous_champion,
        new_champion=new_champion,
        switch_type="auto_demotion",
        effective_date=effective_date,
        reason=f"自动降级: {downgrade_check['reason']}",
        operator="system",
        evaluation_metrics=downgrade_check["metrics_snapshot"],
    )

    # 更新配置
    updated_config = replace(
        config,
        active_champion_id=new_champion,
        champion_source="auto",
        manual_effective_date=effective_date,
        manual_reason=f"自动降级: {downgrade_check['reason']}",
    )

    logger.warning(
        f"Champion 自动降级: {previous_champion} -> {new_champion}, "
        f"原因: {downgrade_check['reason']}, "
        f"触发阈值: {downgrade_check['triggered_thresholds']}"
    )

    return updated_config, audit


# ==================== Challenger 评估报表 ====================

@dataclass
class EvaluationMetrics:
    """评估指标"""

    strategy_id: str
    trade_date: str
    total_return: float = 0.0
    cost_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    turnover: float = 0.0
    win_rate: float = 0.0
    ic: float = 0.0  # Information Coefficient
    ic_ir: float = 0.0  # ICIR
    topk_hit_rate: float = 0.0  # TopK 命中率

    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "trade_date": self.trade_date,
            "total_return": self.total_return,
            "cost_return": self.cost_return,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "turnover": self.turnover,
            "win_rate": self.win_rate,
            "ic": self.ic,
            "ic_ir": self.ic_ir,
            "topk_hit_rate": self.topk_hit_rate,
        }


def calculate_strategy_metrics(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    trade_date: str,
    top_n: int = 10,
    cost_rate: float = 0.003,
) -> EvaluationMetrics:
    """
    计算单策略评估指标

    Args:
        signals: 策略信号 DataFrame（ts_code, score, rank）
        returns: 收益数据 DataFrame（ts_code, return）
        trade_date: 交易日期
        top_n: TopN 数量
        cost_rate: 交易成本率

    Returns:
        评估指标
    """
    if signals.empty or returns.empty:
        return EvaluationMetrics(strategy_id="unknown", trade_date=trade_date)

    # 获取 TopN 信号
    top_signals = signals.nsmallest(top_n, 'rank') if 'rank' in signals.columns else signals.head(top_n)

    # 合并收益
    merged = top_signals.merge(returns, on='ts_code', how='left')

    if merged.empty or 'return' not in merged.columns:
        return EvaluationMetrics(
            strategy_id=signals.get('model_version', ['unknown'])[0] if 'model_version' in signals.columns else 'unknown',
            trade_date=trade_date,
        )

    # 计算指标
    total_return = merged['return'].mean()
    cost_return = total_return - cost_rate * 2  # 买入+卖出成本
    win_rate = (merged['return'] > 0).mean()

    # 计算 IC（信号得分与收益的相关性）
    if 'score' in merged.columns and 'return' in merged.columns:
        ic = merged[['score', 'return']].corr().iloc[0, 1]
    else:
        ic = 0.0

    strategy_id = signals['model_version'].iloc[0] if 'model_version' in signals.columns else 'unknown'

    return EvaluationMetrics(
        strategy_id=strategy_id,
        trade_date=trade_date,
        total_return=float(total_return),
        cost_return=float(cost_return),
        win_rate=float(win_rate),
        ic=float(ic) if not pd.isna(ic) else 0.0,
    )


def generate_challenger_comparison_report(
    champion_signals: pd.DataFrame,
    challenger_signals: Dict[str, pd.DataFrame],
    returns: pd.DataFrame,
    trade_date: str,
    top_n: int = 10,
    cost_rate: float = 0.003,
    champion_id: str = "seed_balance_strategy",
) -> pd.DataFrame:
    """
    生成 Challenger 对比评估报表

    Args:
        champion_signals: Champion 信号
        challenger_signals: Challenger 信号字典
        returns: 收益数据
        trade_date: 交易日期
        top_n: TopN 数量
        cost_rate: 交易成本率
        champion_id: Champion 策略ID

    Returns:
        评估报表 DataFrame
    """
    metrics_list = []

    # 计算 Champion 指标
    champion_metrics = calculate_strategy_metrics(
        signals=champion_signals,
        returns=returns,
        trade_date=trade_date,
        top_n=top_n,
        cost_rate=cost_rate,
    )
    champion_metrics.strategy_id = champion_id
    metrics_list.append(champion_metrics.to_dict())

    # 计算 Challenger 指标
    for strategy_id, signals in challenger_signals.items():
        if signals.empty:
            continue
        metrics = calculate_strategy_metrics(
            signals=signals,
            returns=returns,
            trade_date=trade_date,
            top_n=top_n,
            cost_rate=cost_rate,
        )
        metrics.strategy_id = strategy_id
        metrics_list.append(metrics.to_dict())

    df = pd.DataFrame(metrics_list)

    # 添加相对 Champion 的差异
    if len(df) > 1:
        champion_row = df[df['strategy_id'] == champion_id].iloc[0]
        for col in ['total_return', 'cost_return', 'sharpe', 'ic', 'win_rate']:
            if col in df.columns:
                df[f'{col}_vs_champion'] = df[col] - champion_row[col]

    return df


def generate_multi_period_report(
    metrics_history: pd.DataFrame,
    champion_id: str = "seed_balance_strategy",
    periods: int = 12,
) -> Dict[str, pd.DataFrame]:
    """
    生成多周期评估汇总报表

    Args:
        metrics_history: 历史评估指标 DataFrame
        champion_id: Champion 策略ID
        periods: 统计周期数

    Returns:
        汇总报表字典
    """
    if metrics_history.empty:
        return {"summary": pd.DataFrame(), "trend": pd.DataFrame()}

    # 按策略汇总
    summary = metrics_history.groupby('strategy_id').agg({
        'total_return': ['mean', 'std', 'sum'],
        'cost_return': ['mean', 'std', 'sum'],
        'max_drawdown': ['min', 'max'],
        'sharpe': ['mean', 'std'],
        'turnover': ['mean', 'std'],
        'win_rate': ['mean'],
        'ic': ['mean', 'std'],
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # 添加排名
    summary['return_rank'] = summary['cost_return_mean'].rank(ascending=False)
    summary['sharpe_rank'] = summary['sharpe_mean'].rank(ascending=False)
    summary['ic_rank'] = summary['ic_mean'].rank(ascending=False)
    summary['composite_rank'] = (summary['return_rank'] + summary['sharpe_rank'] + summary['ic_rank']) / 3

    # 计算相对 Champion 的表现
    champion_summary = summary[summary['strategy_id'] == champion_id]
    if not champion_summary.empty:
        for col in ['cost_return_mean', 'sharpe_mean', 'ic_mean']:
            champion_val = champion_summary[col].values[0]
            summary[f'{col}_vs_champion'] = summary[col] - champion_val

    # 趋势分析（最近N期）
    recent = metrics_history.sort_values('trade_date', ascending=False).head(periods * len(metrics_history['strategy_id'].unique()))
    trend = recent.groupby('strategy_id').agg({
        'trade_date': 'count',
        'cost_return': 'mean',
        'sharpe': 'mean',
    }).rename(columns={
        'trade_date': 'period_count',
        'cost_return': 'recent_return',
        'sharpe': 'recent_sharpe',
    }).round(4)

    return {
        "summary": summary,
        "trend": trend.reset_index(),
    }


def save_evaluation_report(
    report_df: pd.DataFrame,
    output_dir: Path,
    trade_date: str,
    report_type: str = "challenger_comparison",
) -> Path:
    """
    保存评估报表

    Args:
        report_df: 报表 DataFrame
        output_dir: 输出目录
        trade_date: 交易日期
        report_type: 报表类型

    Returns:
        保存路径
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{report_type}_{trade_date}.parquet"
    filepath = output_dir / filename

    report_df.to_parquet(filepath, index=False)
    logger.info(f"评估报表已保存: {filepath}")

    return filepath


def format_report_markdown(
    report_df: pd.DataFrame,
    title: str = "Challenger 评估对比报表",
) -> str:
    """
    格式化报表为 Markdown

    Args:
        report_df: 报表 DataFrame
        title: 报表标题

    Returns:
        Markdown 格式的报表
    """
    lines = [
        f"# {title}",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 策略评估指标",
        "",
    ]

    # 表头
    cols = ['strategy_id', 'total_return', 'cost_return', 'win_rate', 'ic', 'sharpe']
    cols = [c for c in cols if c in report_df.columns]

    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"

    lines.append(header)
    lines.append(separator)

    # 数据行
    for _, row in report_df.iterrows():
        values = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")

    # 相对差异表（如果有）
    if 'cost_return_vs_champion' in report_df.columns:
        lines.extend([
            "",
            "## 相对 Champion 差异",
            "",
        ])

        diff_cols = ['strategy_id', 'cost_return_vs_champion', 'sharpe_vs_champion', 'ic_vs_champion']
        diff_cols = [c for c in diff_cols if c in report_df.columns]

        if len(diff_cols) > 1:
            header = "| " + " | ".join(diff_cols) + " |"
            separator = "| " + " | ".join(["---"] * len(diff_cols)) + " |"
            lines.append(header)
            lines.append(separator)

            for _, row in report_df.iterrows():
                values = []
                for col in diff_cols:
                    val = row.get(col, "")
                    if isinstance(val, float):
                        # 差异值用颜色标记
                        if val > 0:
                            values.append(f"+{val:.4f}")
                        else:
                            values.append(f"{val:.4f}")
                    else:
                        values.append(str(val))
                lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


class SeedBalanceStrategy:
    """豆包四因子均衡策略（Champion默认）

    四因子权重体系：
    - 质量因子（30%）：ROIC、毛利率、现金流/净利润、资产负债率
    - 成长因子（30%）：营收增速、净利润增速、PEG、利润加速度
    - 动量因子（20%）：相对行业超额收益、成交额分位数
    - 低波动因子（20%）：波动率、最大回撤、下行波动率
    """

    # 豆包四因子权重
    DOUBAO_WEIGHTS = {
        'quality': 0.30,
        'growth': 0.30,
        'momentum': 0.20,
        'low_vol': 0.20,
    }

    def __init__(
        self,
        selector_config: Optional[SelectionConfig] = None,
        strategy_id: str = "seed_balance_strategy",
        version: str = "v1.0.0",
        use_doubao_scoring: bool = True,
    ):
        if selector_config is None:
            selector_config = SelectionConfig(
                model_type="lgbm",
                label_horizons=(20, 60, 120),
                label_weights=(0.5, 0.3, 0.2),
                risk_adjusted=True,
                industry_col="industry_l1",
            )
        self.selector_config = selector_config
        self.strategy_id = normalize_strategy_id(strategy_id)
        self.model_version = f"{self.strategy_id}@{version}"
        self.selector = StockSelector(self.selector_config)
        self.is_trained = False
        self.use_doubao_scoring = use_doubao_scoring

    def fit(self, train_df: pd.DataFrame) -> None:
        try:
            self.selector.fit(train_df)
            self.is_trained = True
        except ModuleNotFoundError:
            logger.warning("LightGBM不可用，seed_balance_strategy回退到rule模型")
            fallback_config = replace(self.selector_config, model_type="rule")
            self.selector = StockSelector(fallback_config)
            self.selector.fit(train_df)
            self.is_trained = True

    def _calculate_doubao_score(self, df: pd.DataFrame) -> pd.Series:
        """
        计算豆包四因子综合评分

        Args:
            df: 包含各因子评分的 DataFrame

        Returns:
            综合评分 Series
        """
        score = pd.Series(0.0, index=df.index)

        # 质量因子（30%）
        quality_cols = ['quality_score', 'roe_zscore', 'gross_margin_zscore', 'roic_zscore']
        quality_score = self._aggregate_zscore(df, quality_cols)
        score += quality_score * self.DOUBAO_WEIGHTS['quality']

        # 成长因子（30%）
        growth_cols = ['growth_score', 'revenue_yoy_zscore', 'profit_yoy_zscore', 'profit_yoy_accel_zscore']
        growth_score = self._aggregate_zscore(df, growth_cols)
        score += growth_score * self.DOUBAO_WEIGHTS['growth']

        # 动量因子（20%）
        momentum_cols = ['momentum_score', 'excess_return_vs_industry_20d_zscore', 'amt_quantile_zscore']
        momentum_score = self._aggregate_zscore(df, momentum_cols)
        score += momentum_score * self.DOUBAO_WEIGHTS['momentum']

        # 低波动因子（20%）
        low_vol_cols = ['low_vol_score', 'volatility_zscore', 'max_dd_zscore', 'downside_vol_zscore']
        low_vol_score = self._aggregate_zscore(df, low_vol_cols)
        score += low_vol_score * self.DOUBAO_WEIGHTS['low_vol']

        return score

    def _aggregate_zscore(self, df: pd.DataFrame, cols: List[str]) -> pd.Series:
        """
        聚合多个 z-score 列

        Args:
            df: 数据 DataFrame
            cols: 列名列表

        Returns:
            聚合后的评分
        """
        available_cols = [c for c in cols if c in df.columns]
        if not available_cols:
            return pd.Series(0.0, index=df.index)

        # 取均值
        scores = df[available_cols].mean(axis=1)
        return scores.fillna(0)

    def generate_signals(self, data: pd.DataFrame, trade_date: str, top_n: int = 10) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError("seed_balance_strategy 输入数据为空")
        if not self.is_trained:
            self.fit(data)

        predicted = self.selector.predict(data)
        date_col = self.selector_config.date_col
        if date_col in predicted.columns:
            date_norm = pd.to_datetime(predicted[date_col], errors="coerce").dt.strftime("%Y%m%d")
            predicted = predicted[date_norm == _normalize_trade_date(trade_date)]

        # 使用豆包四因子评分（如果启用且有相关特征）
        if self.use_doubao_scoring:
            # 检查是否有豆包评分列
            if 'doubao_champion_score' in predicted.columns:
                score_col = 'doubao_champion_score'
            else:
                # 动态计算豆包评分
                doubao_score = self._calculate_doubao_score(predicted)
                predicted = predicted.copy()
                predicted['doubao_score'] = doubao_score
                score_col = 'doubao_score'
        else:
            score_col = "score"

        return _format_signal_frame(
            raw=predicted,
            score_col=score_col,
            ts_code_col=self.selector_config.code_col,
            trade_date=_normalize_trade_date(trade_date),
            model_version=self.model_version,
            top_n=top_n,
        )


@dataclass
class ChallengerConfig:
    positive_growth_weight: float = 0.7
    positive_frontier_weight: float = 0.3
    # 卫星策略四因子权重（按文档要求）
    satellite_growth_factor_weight: float = 0.40    # 景气成长因子（40%）
    satellite_event_factor_weight: float = 0.30    # 事件驱动因子（30%）
    satellite_momentum_factor_weight: float = 0.20 # 量价动量因子（20%）
    satellite_chip_factor_weight: float = 0.10    # 筹码结构因子（10%）
    # 保留旧权重作为回退（已废弃）
    satellite_growth_weight: float = 0.35
    satellite_frontier_weight: float = 0.25
    satellite_rps_weight: float = 0.20
    satellite_elasticity_weight: float = 0.10
    satellite_not_priced_weight: float = 0.10
    version_map: Dict[str, str] = field(
        default_factory=lambda: {
            "balance_strategy_v1": "v1.0.0",
            "positive_strategy_v1": "v1.0.0",
            "value_strategy_v1": "v1.0.0",
            "satellite_strategy_v1": "v1.0.0",
        }
    )


class MultiAlphaChallengerStrategies:
    """Multi Alpha 拆分后的挑战者策略集合"""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        config: Optional[ChallengerConfig] = None,
        selector: Optional[MultiAlphaStockSelector] = None,
    ):
        self.config = config or ChallengerConfig()
        self.data_dir = data_dir
        self.selector = selector or MultiAlphaStockSelector(data_dir=data_dir)

    def generate_signals(
        self,
        trade_date: str,
        top_n: int = 30,
        allocation_method: str = "fixed",
        regime: str = "sideways",
    ) -> Dict[str, pd.DataFrame]:
        def _normalize_weights(weights: Dict[str, float], fallback: Dict[str, float]) -> Dict[str, float]:
            cleaned = {}
            for key, val in weights.items():
                try:
                    cleaned[key] = float(val)
                except (TypeError, ValueError):
                    cleaned[key] = 0.0
            total = sum(v for v in cleaned.values() if np.isfinite(v) and v > 0)
            if total <= 0:
                return fallback.copy()
            return {k: max(v, 0.0) / total for k, v in cleaned.items()}

        def _safe_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
            if column not in frame.columns:
                return pd.Series(0.0, index=frame.index, dtype=float)
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

        result = self.selector.select(
            trade_date=trade_date,
            top_n=max(top_n, 30),
            allocation_method=allocation_method,
            regime=regime,
        )
        all_scores = result.get("all_scores")
        if all_scores is None or all_scores.empty:
            return {
                "balance_strategy_v1": pd.DataFrame(columns=SIGNAL_SCHEMA),
                "positive_strategy_v1": pd.DataFrame(columns=SIGNAL_SCHEMA),
                "value_strategy_v1": pd.DataFrame(columns=SIGNAL_SCHEMA),
                "satellite_strategy_v1": pd.DataFrame(columns=SIGNAL_SCHEMA),
            }

        data = all_scores.copy()
        positive_weights = _normalize_weights(
            {
                "growth": self.config.positive_growth_weight,
                "frontier": self.config.positive_frontier_weight,
            },
            {"growth": 0.7, "frontier": 0.3},
        )

        data["positive_score"] = (
            _safe_numeric(data, "growth_score") * positive_weights["growth"]
            + _safe_numeric(data, "frontier_score") * positive_weights["frontier"]
        )

        satellite_weights = _normalize_weights(
            {
                "growth_factor": self.config.satellite_growth_factor_weight,
                "event_factor": self.config.satellite_event_factor_weight,
                "momentum_factor": self.config.satellite_momentum_factor_weight,
                "chip_factor": self.config.satellite_chip_factor_weight,
            },
            {
                "growth_factor": 0.40,   # 景气成长因子
                "event_factor": 0.30,    # 事件驱动因子
                "momentum_factor": 0.20, # 量价动量因子
                "chip_factor": 0.10,    # 筹码结构因子
            },
        )

        # 卫星策略四因子评分（按文档要求）
        # 如果有 satellite_score（由 SatelliteFeatures 计算），直接使用
        # 否则使用旧逻辑回退
        if "satellite_score" in data.columns:
            # 使用 SatelliteFeatures 计算的完整评分
            pass
        else:
            # 回退：使用简化评分
            data["satellite_score"] = (
                _safe_numeric(data, "growth_score") * satellite_weights["growth_factor"]
                + _safe_numeric(data, "frontier_score") * satellite_weights["event_factor"]
                + _safe_numeric(data, "momentum_factor_score", 
                               _safe_numeric(data, "rps_component")) * satellite_weights["momentum_factor"]
                + _safe_numeric(data, "chip_factor_score",
                               _safe_numeric(data, "not_priced_component")) * satellite_weights["chip_factor"]
            )

        outputs = {}
        score_map = {
            "balance_strategy_v1": "combined_score",
            "positive_strategy_v1": "positive_score",
            "value_strategy_v1": "value_score",
            "satellite_strategy_v1": "satellite_score",
        }
        for strategy_id, score_col in score_map.items():
            version = self.config.version_map.get(strategy_id, "v1.0.0")
            outputs[strategy_id] = _format_signal_frame(
                raw=data,
                score_col=score_col,
                ts_code_col="ts_code",
                trade_date=trade_date,
                model_version=f"{strategy_id}@{version}",
                top_n=top_n,
            )
        return outputs


class ChampionChallengerEngine:
    """冠军/挑战者运行引擎"""

    def __init__(
        self,
        governance_config: Optional[StrategyGovernanceConfig] = None,
        seed_strategy: Optional[SeedBalanceStrategy] = None,
        challenger_strategies: Optional[MultiAlphaChallengerStrategies] = None,
    ):
        self.governance_config = governance_config or StrategyGovernanceConfig()
        self.seed_strategy = seed_strategy or SeedBalanceStrategy()
        self.challenger_strategies = challenger_strategies or MultiAlphaChallengerStrategies()

    def run(
        self,
        trade_date: str,
        top_n: int = 10,
        seed_data: Optional[pd.DataFrame] = None,
        active_champion_id: Optional[str] = None,
        allocation_method: str = "fixed",
        regime: str = "sideways",
    ) -> Dict[str, object]:
        champion_id = normalize_strategy_id(
            active_champion_id or self.governance_config.normalized_active_champion_id()
        )

        try:
            challengers = self.challenger_strategies.generate_signals(
                trade_date=trade_date,
                top_n=top_n,
                allocation_method=allocation_method,
                regime=regime,
            )
        except Exception as exc:
            logger.warning("挑战者策略生成失败，已降级为空输出: %s", exc)
            challengers = {
                "balance_strategy_v1": pd.DataFrame(columns=SIGNAL_SCHEMA),
                "positive_strategy_v1": pd.DataFrame(columns=SIGNAL_SCHEMA),
                "value_strategy_v1": pd.DataFrame(columns=SIGNAL_SCHEMA),
                "satellite_strategy_v1": pd.DataFrame(columns=SIGNAL_SCHEMA),
            }

        if champion_id == "seed_balance_strategy":
            if seed_data is None or seed_data.empty:
                raise ValueError("active_champion_id=seed_balance_strategy 时必须提供 seed_data")
            champion_signal = self.seed_strategy.generate_signals(
                data=seed_data,
                trade_date=trade_date,
                top_n=top_n,
            )
        else:
            champion_signal = challengers.get(champion_id)
            if champion_signal is None:
                raise ValueError(f"无法获取冠军策略输出: {champion_id}")

        return {
            "trade_date": _normalize_trade_date(trade_date),
            "active_champion_id": champion_id,
            "champion_signals": champion_signal,
            "challenger_signals": challengers,
        }


def save_strategy_outputs(
    output_root: Path,
    trade_date: str,
    champion_id: str,
    champion_signals: pd.DataFrame,
    challenger_signals: Dict[str, pd.DataFrame],
) -> Dict[str, Path]:
    output_root = Path(output_root)
    champion_dir = output_root / "champion"
    challenger_dir = output_root / "challenger"
    champion_dir.mkdir(parents=True, exist_ok=True)
    challenger_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}
    champion_path = champion_dir / f"{trade_date}.parquet"
    champion_signals.to_parquet(champion_path, index=False)
    paths[f"champion:{champion_id}"] = champion_path

    for strategy_id, frame in challenger_signals.items():
        strategy_dir = challenger_dir / strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)
        path = strategy_dir / f"{trade_date}.parquet"
        frame.to_parquet(path, index=False)
        paths[f"challenger:{strategy_id}"] = path

    return paths
