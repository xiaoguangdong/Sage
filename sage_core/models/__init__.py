"""
模型模块
"""
from .trend_model import TrendModelRule, TrendModelLGBM, TrendModelHMM, create_trend_model
from .entry_model import EntryModelLR
from .stock_selector import StockSelector, SelectionConfig
from .strategy_governance import (
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
    from .rank_model import RankModelLGBM
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
