"""
特征模块
"""
from .base import FeatureGenerator, FeatureSpec
from .registry import FEATURE_REGISTRY, FeatureRegistry, register_feature
from .price_features import PriceFeatures
from .market_features import MarketFeatures
from .fundamental_features import FundamentalFeatures
from .flow_features import FlowFeatures
from .industry_features import IndustryFeatures
from .pipeline import FeaturePipeline, FeaturePipelineResult

__all__ = [
    "FeatureGenerator",
    "FeatureSpec",
    "FeatureRegistry",
    "FEATURE_REGISTRY",
    "register_feature",
    "PriceFeatures",
    "MarketFeatures",
    "FundamentalFeatures",
    "FlowFeatures",
    "IndustryFeatures",
    "FeaturePipeline",
    "FeaturePipelineResult",
]
