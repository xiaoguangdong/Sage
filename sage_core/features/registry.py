from __future__ import annotations

from typing import Dict, Type

from .base import FeatureGenerator


class FeatureRegistry:
    """
    特征生成器注册表
    """

    def __init__(self):
        self._registry: Dict[str, Type[FeatureGenerator]] = {}

    def register(self, generator_cls: Type[FeatureGenerator]) -> None:
        name = getattr(generator_cls, "spec", None).name if hasattr(generator_cls, "spec") else None
        if not name:
            raise ValueError("FeatureGenerator missing spec.name")
        if name in self._registry:
            raise ValueError(f"Feature already registered: {name}")
        self._registry[name] = generator_cls

    def get(self, name: str) -> Type[FeatureGenerator]:
        if name not in self._registry:
            raise KeyError(f"Unknown feature: {name}")
        return self._registry[name]

    def create(self, name: str, **kwargs) -> FeatureGenerator:
        return self.get(name)(**kwargs)

    def list(self) -> Dict[str, Type[FeatureGenerator]]:
        return dict(self._registry)


FEATURE_REGISTRY = FeatureRegistry()


def register_feature(cls: Type[FeatureGenerator]) -> Type[FeatureGenerator]:
    FEATURE_REGISTRY.register(cls)
    return cls
