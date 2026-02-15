"""
模型兼容层（仅保留历史导入路径）

新代码请直接使用按业务域拆分后的模块：
- sage_core.trend
- sage_core.industry
- sage_core.stock_selection
- sage_core.execution
- sage_core.governance
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
