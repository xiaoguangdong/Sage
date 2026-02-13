from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from .base import FeatureGenerator
from .registry import FEATURE_REGISTRY
from sage_core.data import DataStore, DataCatalog, DatasetSpec


@dataclass
class FeaturePipelineResult:
    pipeline: str
    input_columns: List[str]
    feature_columns: List[str]
    df: pd.DataFrame


class FeaturePipeline:
    """
    特征流水线
    """

    def __init__(self, generators: Sequence[FeatureGenerator], name: str = "feature_pipeline"):
        if not generators:
            raise ValueError("FeaturePipeline requires at least one generator")
        self.generators = list(generators)
        self.name = name

    @classmethod
    def from_names(cls, names: Iterable[str], name: str = "feature_pipeline", **kwargs):
        generators = [FEATURE_REGISTRY.create(n, **kwargs) for n in names]
        return cls(generators, name=name)

    def run(
        self,
        df: pd.DataFrame,
        store: Optional[DataStore] = None,
        dataset_name: Optional[str] = None,
    ) -> FeaturePipelineResult:
        input_columns = list(df.columns)
        result_df = df.copy()

        for generator in self.generators:
            result_df = generator.transform(result_df)

        feature_columns = [c for c in result_df.columns if c not in input_columns]

        if store and dataset_name:
            store.write_parquet(dataset_name, result_df)

        return FeaturePipelineResult(
            pipeline=self.name,
            input_columns=input_columns,
            feature_columns=feature_columns,
            df=result_df,
        )
