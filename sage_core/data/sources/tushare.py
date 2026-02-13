from __future__ import annotations

import pandas as pd

from .base import DataSource, SourceConfig, register_source


@register_source
class TushareSource(DataSource):
    name = "tushare"

    def __init__(self, config: SourceConfig):
        super().__init__(config)

    def available_datasets(self):
        return [
            "daily_basic",
            "margin",
            "daily_kline",
            "index_ohlc",
            "hs300_constituents",
            "hs300_moneyflow",
            "sw_industry_classify",
            "sw_industry_daily",
            "opt_daily",
            "fina_indicator",
            "fina_indicator_vip",
            "tushare_sectors",
            "concept_update_tushare",
        ]

    def fetch(self, dataset: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("TushareSource 仅提供结构骨架，数据拉取请使用 scripts/data/tushare_suite.py")
