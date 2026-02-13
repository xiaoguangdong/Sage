"""
回测模块
"""
from .types import BacktestConfig, BacktestResult
from .engine import BacktestEngine
from .simple_engine import SimpleBacktestEngine
from .walk_forward import WalkForwardBacktest

__all__ = ["BacktestConfig", "BacktestResult", "BacktestEngine", "WalkForwardBacktest", "SimpleBacktestEngine"]
