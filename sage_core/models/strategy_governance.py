from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from .stock_selector import SelectionConfig, StockSelector
from .multi_alpha_selector import MultiAlphaStockSelector

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
    )

    def normalized_active_champion_id(self) -> str:
        return normalize_strategy_id(self.active_champion_id)

    def normalized_challengers(self) -> Tuple[str, ...]:
        return tuple(normalize_strategy_id(s) for s in self.challengers)


class SeedBalanceStrategy:
    """豆包硬指标均衡策略（Champion默认）"""

    def __init__(
        self,
        selector_config: Optional[SelectionConfig] = None,
        strategy_id: str = "seed_balance_strategy",
        version: str = "v1.0.0",
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
        return _format_signal_frame(
            raw=predicted,
            score_col="score",
            ts_code_col=self.selector_config.code_col,
            trade_date=_normalize_trade_date(trade_date),
            model_version=self.model_version,
            top_n=top_n,
        )


@dataclass
class ChallengerConfig:
    positive_growth_weight: float = 0.7
    positive_frontier_weight: float = 0.3
    version_map: Dict[str, str] = field(
        default_factory=lambda: {
            "balance_strategy_v1": "v1.0.0",
            "positive_strategy_v1": "v1.0.0",
            "value_strategy_v1": "v1.0.0",
        }
    )


class MultiAlphaChallengerStrategies:
    """Multi Alpha 拆分后的三个挑战者策略"""

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
            }

        data = all_scores.copy()
        growth_w = float(self.config.positive_growth_weight)
        frontier_w = float(self.config.positive_frontier_weight)
        weight_sum = growth_w + frontier_w
        if weight_sum <= 0:
            growth_w, frontier_w = 0.7, 0.3
            weight_sum = 1.0
        growth_w = growth_w / weight_sum
        frontier_w = frontier_w / weight_sum

        data["positive_score"] = (
            pd.to_numeric(data["growth_score"], errors="coerce").fillna(0.0) * growth_w
            + pd.to_numeric(data["frontier_score"], errors="coerce").fillna(0.0) * frontier_w
        )

        outputs = {}
        score_map = {
            "balance_strategy_v1": "combined_score",
            "positive_strategy_v1": "positive_score",
            "value_strategy_v1": "value_score",
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
