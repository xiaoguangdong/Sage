from __future__ import annotations

from typing import Iterable, List

import pandas as pd


def compute_factor_exposures(
    df: pd.DataFrame,
    factor_cols: Iterable[str],
    *,
    date_col: str = "trade_date",
    code_col: str = "ts_code",
    industry_col: str = "industry_l1",
    zscore: bool = True,
    neutralize_industry: bool = True,
) -> pd.DataFrame:
    factor_cols = [c for c in factor_cols if c in df.columns]
    if not factor_cols:
        raise ValueError("未找到可用因子列")

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.date
    for col in factor_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if neutralize_industry and industry_col in data.columns:
        for col in factor_cols:
            data[col] = data.groupby([date_col, industry_col])[col].transform(lambda x: x - x.mean())

    if zscore:
        for col in factor_cols:
            data[col] = data.groupby(date_col)[col].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8))

    output_cols: List[str] = [date_col, code_col]
    if industry_col in data.columns:
        output_cols.append(industry_col)
    output_cols.extend(factor_cols)
    return data[output_cols]
