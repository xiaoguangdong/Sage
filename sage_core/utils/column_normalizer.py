"""
统一字段规范工具
"""
from __future__ import annotations

import pandas as pd


CODE_CANDIDATES = ("ts_code", "code", "stock")
DATE_CANDIDATES = ("trade_date", "date", "datetime")
INDUSTRY_CANDIDATES = ("sw_industry", "industry", "industry_name", "industry_l1")


def normalize_ts_code(value: str) -> str:
    if value is None:
        return value
    code = str(value).strip()
    if not code:
        return code
    lower = code.lower()
    if lower.startswith("sh.") or lower.startswith("sz."):
        suffix = lower[:2].upper()
        return f"{code[3:]}.{suffix}"
    if lower.endswith(".sh") or lower.endswith(".sz"):
        return f"{code[:-3]}.{code[-2:].upper()}"
    return code.upper() if "." in code else code


def normalize_trade_date(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y%m%d")
    sample = series.dropna().astype(str)
    if sample.empty:
        return series
    sample_str = sample.iloc[0]
    if sample_str.isdigit() and len(sample_str) in (6, 8):
        return series.astype(str).str.zfill(len(sample_str))
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        return parsed.dt.strftime("%Y%m%d")
    return series.astype(str)


def normalize_security_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    统一股票代码与日期字段：
    - 确保 ts_code 与 trade_date 存在
    - 保留/补齐 code 与 date 兼容旧代码
    """
    target = df if inplace else df.copy()

    if "ts_code" not in target.columns:
        for col in CODE_CANDIDATES:
            if col in target.columns:
                target["ts_code"] = target[col].map(normalize_ts_code)
                break

    if "trade_date" not in target.columns:
        for col in DATE_CANDIDATES:
            if col in target.columns:
                target["trade_date"] = normalize_trade_date(target[col])
                break

    if "code" not in target.columns and "ts_code" in target.columns:
        target["code"] = target["ts_code"]
    if "stock" not in target.columns and "ts_code" in target.columns:
        target["stock"] = target["ts_code"]
    if "date" not in target.columns and "trade_date" in target.columns:
        target["date"] = target["trade_date"]

    return target


def normalize_industry_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    统一行业字段：
    - 确保 sw_industry 存在
    - 保留/补齐 industry 与 industry_name 兼容旧代码
    """
    target = df if inplace else df.copy()

    if "sw_industry" not in target.columns:
        for col in INDUSTRY_CANDIDATES:
            if col in target.columns and col != "sw_industry":
                target["sw_industry"] = target[col]
                break

    if "industry_name" not in target.columns and "sw_industry" in target.columns:
        target["industry_name"] = target["sw_industry"]
    if "industry" not in target.columns and "sw_industry" in target.columns:
        target["industry"] = target["sw_industry"]

    return target
