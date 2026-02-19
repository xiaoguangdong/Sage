"""
特征模块
"""

from .base import FeatureGenerator, FeatureSpec
from .flow_features import FlowFeatures
from .fundamental_features import FundamentalFeatures
from .industry_features import IndustryFeatures
from .market_features import MarketFeatures
from .pipeline import FeaturePipeline, FeaturePipelineResult
from .price_features import PriceFeatures
from .registry import FEATURE_REGISTRY, FeatureRegistry, register_feature
from .satellite_features import SatelliteFeatures

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
    "SatelliteFeatures",
    "FeaturePipeline",
    "FeaturePipelineResult",
]
