from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from scripts.data._shared.runtime import get_data_path


@dataclass(frozen=True)
class DatasetSpec:
    """
    数据集描述

    name: 数据集唯一名称
    section: 数据分区（raw/processed/features/labels/backtest/cache/meta）
    relative_path: 相对路径（基于 section 下的路径）
    description: 人类可读说明
    """

    name: str
    section: str
    relative_path: str
    description: str = ""


class DataCatalog:
    """
    数据目录索引与读取入口

    - 默认优先读取 primary 数据根
    - primary 不存在则回退到 secondary
    - 写入默认走 primary（可通过 root_kind 参数指定）
    """

    def __init__(self, specs: Optional[Iterable[DatasetSpec]] = None):
        self._specs: Dict[str, DatasetSpec] = {}
        for spec in (specs or self._default_specs()):
            self.register(spec)

    def register(self, spec: DatasetSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Dataset already registered: {spec.name}")
        self._specs[spec.name] = spec

    def list_specs(self) -> Dict[str, DatasetSpec]:
        return dict(self._specs)

    def get_spec(self, name: str) -> DatasetSpec:
        if name not in self._specs:
            raise KeyError(f"Unknown dataset: {name}")
        return self._specs[name]

    def get_path(self, name: str, root_kind: str = "primary", ensure: bool = False) -> Path:
        spec = self.get_spec(name)
        return get_data_path(spec.section, spec.relative_path, root_kind=root_kind, ensure=ensure)

    def resolve_path(self, name: str) -> Path:
        """
        获取可读路径（优先 primary，若不存在则回退 secondary）
        """
        primary = self.get_path(name, root_kind="primary", ensure=False)
        if primary.exists():
            return primary
        secondary = self.get_path(name, root_kind="secondary", ensure=False)
        if secondary.exists():
            return secondary
        return primary

    def exists(self, name: str) -> bool:
        path = self.resolve_path(name)
        return path.exists()

    def read_parquet(self, name: str, **kwargs) -> pd.DataFrame:
        path = self.resolve_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {name} -> {path}")
        return pd.read_parquet(path, **kwargs)

    def write_parquet(self, name: str, df: pd.DataFrame, **kwargs) -> Path:
        path = self.get_path(name, root_kind="primary", ensure=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, **kwargs)
        return path

    @staticmethod
    def _default_specs() -> Iterable[DatasetSpec]:
        return [
            DatasetSpec(
                name="index_ohlc_all",
                section="raw",
                relative_path="tushare/index/index_ohlc_all.parquet",
                description="指数汇总OHLC（日频）",
            ),
            DatasetSpec(
                name="index_000300",
                section="raw",
                relative_path="tushare/index/index_000300_SH_ohlc.parquet",
                description="沪深300指数OHLC",
            ),
            DatasetSpec(
                name="daily_kline_dir",
                section="raw",
                relative_path="tushare/daily",
                description="个股日线K线分年目录",
            ),
            DatasetSpec(
                name="daily_basic_parts_dir",
                section="raw",
                relative_path="tushare/daily_basic/parts",
                description="daily_basic 分片目录",
            ),
            DatasetSpec(
                name="margin_parts_dir",
                section="raw",
                relative_path="tushare/margin/parts",
                description="融资融券分片目录",
            ),
            DatasetSpec(
                name="hs300_constituents",
                section="raw",
                relative_path="tushare/constituents/hs300_constituents_all.parquet",
                description="沪深300成分股",
            ),
            DatasetSpec(
                name="concept_details",
                section="raw",
                relative_path="tushare/sectors/all_concept_details.csv",
                description="概念成分股明细",
            ),
            DatasetSpec(
                name="factor_scores",
                section="processed",
                relative_path="factors/stock_factors_with_score.parquet",
                description="因子得分",
            ),
        ]
