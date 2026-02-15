"""
模型兼容层（向后兼容）
"""
from sage_core.trend.trend_model import TrendModelRule, TrendModelLGBM, TrendModelHMM, create_trend_model
from sage_core.execution.entry_model import EntryModelLR
from sage_core.stock_selection.stock_selector import StockSelector, SelectionConfig
from sage_core.governance.strategy_governance import (
    SIGNAL_SCHEMA,
    SUPPORTED_STRATEGIES,
    StrategyGovernanceConfig,
    SeedBalanceStrategy,
    ChallengerConfig,
    MultiAlphaChallengerStrategies,
    ChampionChallengerEngine,
    normalize_strategy_id,
    decide_auto_promotion,
)

try:
    from sage_core.stock_selection.rank_model import RankModelLGBM
except ModuleNotFoundError:
    RankModelLGBM = None

__all__ = [
    'TrendModelRule',
    'TrendModelLGBM',
    'TrendModelHMM',
    'create_trend_model',
    'RankModelLGBM',
    'EntryModelLR',
    'StockSelector',
    'SelectionConfig',
    'SIGNAL_SCHEMA',
    'SUPPORTED_STRATEGIES',
    'StrategyGovernanceConfig',
    'SeedBalanceStrategy',
    'ChallengerConfig',
    'MultiAlphaChallengerStrategies',
    'ChampionChallengerEngine',
    'normalize_strategy_id',
    'decide_auto_promotion',
]
