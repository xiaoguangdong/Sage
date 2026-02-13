from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .catalog import DataCatalog, DatasetSpec


class DataStore:
    """
    统一数据访问入口
    """

    def __init__(self, catalog: Optional[DataCatalog] = None):
        self.catalog = catalog or DataCatalog()

    def list_specs(self) -> Dict[str, DatasetSpec]:
        return self.catalog.list_specs()

    def path(self, name: str, root_kind: str = "primary", ensure: bool = False) -> Path:
        return self.catalog.get_path(name, root_kind=root_kind, ensure=ensure)

    def resolve(self, name: str) -> Path:
        return self.catalog.resolve_path(name)

    def exists(self, name: str) -> bool:
        return self.catalog.exists(name)

    def read_parquet(self, name: str, **kwargs) -> pd.DataFrame:
        return self.catalog.read_parquet(name, **kwargs)

    def write_parquet(self, name: str, df: pd.DataFrame, **kwargs) -> Path:
        return self.catalog.write_parquet(name, df, **kwargs)

    def read_csv(self, name: str, **kwargs) -> pd.DataFrame:
        path = self.resolve(name)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {name} -> {path}")
        return pd.read_csv(path, **kwargs)

    def write_csv(self, name: str, df: pd.DataFrame, **kwargs) -> Path:
        path = self.path(name, root_kind="primary", ensure=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, **kwargs)
        return path
