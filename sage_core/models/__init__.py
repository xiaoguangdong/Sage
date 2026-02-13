"""
模型模块
"""
from .trend_model import TrendModelRule, TrendModelLGBM, TrendModelHMM, create_trend_model
from .entry_model import EntryModelLR
from .stock_selector import StockSelector, SelectionConfig

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
    'SelectionConfig'
]
