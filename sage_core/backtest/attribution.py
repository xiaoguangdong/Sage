from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class BrinsonAttributionResult:
    by_industry: pd.DataFrame
    by_date: pd.DataFrame


@dataclass
class FactorAttributionResult:
    factor_returns: pd.DataFrame
    factor_contributions: pd.DataFrame


def _weighted_avg(df: pd.DataFrame, weight_col: str, value_col: str) -> float:
    weights = df[weight_col].fillna(0.0).to_numpy()
    if weights.sum() == 0:
        return 0.0
    values = df[value_col].fillna(0.0).to_numpy()
    return float(np.dot(weights, values) / weights.sum())


def brinson_attribution(
    portfolio: pd.DataFrame,
    benchmark: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    industry_col: str = "industry_l1",
    weight_col: str = "weight",
    return_col: str = "return",
) -> BrinsonAttributionResult:
    """
    Brinson-Fachler 归因（配置效应/选股效应/交互效应）
    输入需要包含: date_col, industry_col, weight_col, return_col
    """
    required = {date_col, industry_col, weight_col, return_col}
    if not required.issubset(portfolio.columns) or not required.issubset(benchmark.columns):
        raise ValueError("portfolio/benchmark缺少必要列")

    pf = portfolio.copy()
    bm = benchmark.copy()
    pf[date_col] = pd.to_datetime(pf[date_col]).dt.date
    bm[date_col] = pd.to_datetime(bm[date_col]).dt.date

    pf_group = (
        pf.groupby([date_col, industry_col], dropna=False)
        .apply(lambda x: pd.Series({"Wp": x[weight_col].sum(), "Rp": _weighted_avg(x, weight_col, return_col)}))
        .reset_index()
    )
    bm_group = (
        bm.groupby([date_col, industry_col], dropna=False)
        .apply(lambda x: pd.Series({"Wb": x[weight_col].sum(), "Rb": _weighted_avg(x, weight_col, return_col)}))
        .reset_index()
    )

    merged = pd.merge(pf_group, bm_group, on=[date_col, industry_col], how="outer").fillna(0.0)
    rb_total = merged.groupby(date_col).apply(lambda x: float((x["Wb"] * x["Rb"]).sum())).rename("Rb_total")
    merged = merged.merge(rb_total, left_on=date_col, right_index=True)

    merged["allocation"] = (merged["Wp"] - merged["Wb"]) * (merged["Rb"] - merged["Rb_total"])
    merged["selection"] = merged["Wb"] * (merged["Rp"] - merged["Rb"])
    merged["interaction"] = (merged["Wp"] - merged["Wb"]) * (merged["Rp"] - merged["Rb"])

    by_date = merged.groupby(date_col)[["allocation", "selection", "interaction"]].sum().reset_index()
    by_date["excess_return"] = by_date["allocation"] + by_date["selection"] + by_date["interaction"]

    return BrinsonAttributionResult(by_industry=merged, by_date=by_date)


def factor_attribution(
    data: pd.DataFrame,
    factor_cols: Iterable[str],
    *,
    date_col: str = "trade_date",
    weight_col: str = "weight",
    return_col: str = "return",
) -> FactorAttributionResult:
    """
    简易因子归因：按日期做加权最小二乘回归，输出因子收益与组合贡献。
    """
    factor_cols = list(factor_cols)
    required = {date_col, weight_col, return_col, *factor_cols}
    if not required.issubset(data.columns):
        raise ValueError("data缺少必要列")

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    factor_returns: List[pd.DataFrame] = []
    contributions: List[pd.DataFrame] = []

    for dt, group in df.groupby(date_col):
        X = group[factor_cols].fillna(0.0).to_numpy()
        y = group[return_col].fillna(0.0).to_numpy()
        w = group[weight_col].fillna(0.0).to_numpy()

        if w.sum() == 0:
            continue
        w = w / w.sum()
        W = np.diag(w)
        X_design = np.column_stack([np.ones(len(group)), X])

        try:
            beta = np.linalg.pinv(X_design.T @ W @ X_design) @ (X_design.T @ W @ y)
        except np.linalg.LinAlgError:
            continue

        factor_ret = pd.Series(beta[1:], index=factor_cols)
        factor_returns.append(pd.DataFrame({"trade_date": dt, **factor_ret.to_dict()}))

        exposures = (group[factor_cols].fillna(0.0).multiply(w, axis=0)).sum().to_dict()
        contrib = {k: exposures.get(k, 0.0) * factor_ret.get(k, 0.0) for k in factor_cols}
        contributions.append(pd.DataFrame({"trade_date": dt, **contrib}))

    factor_returns_df = pd.concat(factor_returns, ignore_index=True) if factor_returns else pd.DataFrame()
    factor_contrib_df = pd.concat(contributions, ignore_index=True) if contributions else pd.DataFrame()
    return FactorAttributionResult(factor_returns=factor_returns_df, factor_contributions=factor_contrib_df)
