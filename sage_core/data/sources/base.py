from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable

import pandas as pd


@dataclass
class SourceConfig:
    name: str
    params: Dict[str, str] = field(default_factory=dict)


class DataSource:
    """
    数据源抽象基类
    """

    name: str = "unknown"

    def __init__(self, config: SourceConfig):
        self.config = config

    def available_datasets(self) -> Iterable[str]:
        return []

    def fetch(self, dataset: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def supports(self, dataset: str) -> bool:
        return dataset in set(self.available_datasets())


class DataSourceRegistry:
    def __init__(self):
        self._registry: Dict[str, type[DataSource]] = {}

    def register(self, source_cls: type[DataSource]) -> None:
        name = getattr(source_cls, "name", None)
        if not name:
            raise ValueError("DataSource missing name")
        if name in self._registry:
            raise ValueError(f"DataSource already registered: {name}")
        self._registry[name] = source_cls

    def get(self, name: str) -> type[DataSource]:
        if name not in self._registry:
            raise KeyError(f"Unknown data source: {name}")
        return self._registry[name]

    def create(self, name: str, config: SourceConfig) -> DataSource:
        return self.get(name)(config)

    def list(self) -> Dict[str, type[DataSource]]:
        return dict(self._registry)


SOURCE_REGISTRY = DataSourceRegistry()


def register_source(cls: type[DataSource]) -> type[DataSource]:
    SOURCE_REGISTRY.register(cls)
    return cls
