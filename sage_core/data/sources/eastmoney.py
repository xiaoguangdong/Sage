from __future__ import annotations

import pandas as pd

from .base import DataSource, SourceConfig, register_source


@register_source
class EastmoneySource(DataSource):
    name = "eastmoney"

    def __init__(self, config: SourceConfig):
        super().__init__(config)

    def available_datasets(self):
        return [
            "concept_update_eastmoney",
            "concepts",
        ]

    def fetch(self, dataset: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("EastmoneySource 仅提供结构骨架，后续实现")
