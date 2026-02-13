from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    """
    特征生成器描述

    name: 唯一名称
    input_fields: 必需输入字段
    output_fields: 预期输出字段（可为空，生成器自说明）
    description: 功能说明
    version: 版本标识
    """

    name: str
    input_fields: Tuple[str, ...] = ()
    output_fields: Tuple[str, ...] = ()
    description: str = ""
    version: str = "v1"


class FeatureGenerator:
    """
    特征生成器基类
    """

    spec: FeatureSpec

    def fit(self, df: pd.DataFrame) -> "FeatureGenerator":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def validate_input(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.spec.input_fields if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必要字段: {missing}")

    @staticmethod
    def _as_tuple(values: Iterable[str]) -> Tuple[str, ...]:
        return tuple(values)
