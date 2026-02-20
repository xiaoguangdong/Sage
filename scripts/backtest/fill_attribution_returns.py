#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

try:
    from pyarrow import dataset as ds  # type: ignore
except Exception:  # pragma: no cover
    ds = None

from scripts.data._shared.runtime import get_data_path, get_tushare_root, setup_logger

logger = setup_logger("fill_attribution_returns", module="backtest")


def _load_trading_calendar(index_path: Path) -> List[str]:
    if not index_path.exists():
        return []
    df = pd.read_parquet(index_path, columns=["date"])
    dates = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y%m%d")
    return sorted(set(dates.dropna().tolist()))


def _build_next_map(dates: List[str], horizon: int) -> dict[str, Optional[str]]:
    mapping: dict[str, Optional[str]] = {}
    for i, d in enumerate(dates):
        j = i + horizon
        mapping[d] = dates[j] if j < len(dates) else None
    return mapping


def _load_prices_daily(tushare_root: Path, dates: Iterable[str]) -> pd.DataFrame:
    path = tushare_root / "daily.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    date_list = sorted(set(dates))
    if not date_list:
        return pd.DataFrame(columns=["ts_code", "trade_date", "close"])
    if ds is not None:
        dataset = ds.dataset(str(path), format="parquet")
        table = dataset.to_table(
            columns=["ts_code", "trade_date", "close"],
            filter=ds.field("trade_date").isin(date_list),
        )
        df = table.to_pandas()
    else:
        df = pd.read_parquet(path, columns=["ts_code", "trade_date", "close"])
        df = df[df["trade_date"].astype(str).isin(date_list)]
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def _load_prices_sw_industry(tushare_root: Path, dates: Iterable[str]) -> pd.DataFrame:
    path = tushare_root / "sectors" / "sw_daily_all.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["name", "trade_date", "close"])
    date_list = sorted(set(dates))
    if not date_list:
        return pd.DataFrame(columns=["name", "trade_date", "close"])
    if ds is not None:
        dataset = ds.dataset(str(path), format="parquet")
        table = dataset.to_table(
            columns=["name", "trade_date", "close"],
            filter=ds.field("trade_date").isin(date_list),
        )
        df = table.to_pandas()
    else:
        df = pd.read_parquet(path, columns=["name", "trade_date", "close"])
        df = df[df["trade_date"].astype(str).isin(date_list)]
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def _fill_stock_returns(df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    price_df = price_df.rename(columns={"close": "close_t0"})
    df = df.merge(price_df, on=["ts_code", "trade_date"], how="left")
    price_next = price_df.rename(columns={"trade_date": "next_trade_date", "close_t0": "close_t1"})
    df = df.merge(price_next, on=["ts_code", "next_trade_date"], how="left")
    mask = df["return"].isna()
    df.loc[mask, "return"] = df.loc[mask, "close_t1"] / df.loc[mask, "close_t0"] - 1.0
    return df.drop(columns=["close_t0", "close_t1"])


def _fill_industry_returns(df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    price_df = price_df.rename(columns={"close": "close_t0", "name": "industry_l1"})
    df = df.merge(price_df, on=["industry_l1", "trade_date"], how="left")
    price_next = price_df.rename(columns={"trade_date": "next_trade_date", "close_t0": "close_t1"})
    df = df.merge(price_next, on=["industry_l1", "next_trade_date"], how="left")
    mask = df["return"].isna()
    df.loc[mask, "return"] = df.loc[mask, "close_t1"] / df.loc[mask, "close_t0"] - 1.0
    return df.drop(columns=["close_t0", "close_t1"])


def fill_returns(input_dir: Path, horizon: int) -> None:
    tushare_root = get_tushare_root()
    calendar_path = tushare_root / "index" / "index_ohlc_all.parquet"
    calendar = _load_trading_calendar(calendar_path)
    if not calendar:
        raise RuntimeError("缺少交易日历数据（index_ohlc_all.parquet）")
    next_map = _build_next_map(calendar, horizon)

    factor_files = sorted(input_dir.glob("factor_exposure_*.csv"))
    industry_files = sorted(input_dir.glob("portfolio_industry_*.csv"))

    for path in factor_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["trade_date"] = df["trade_date"].astype(str)
        df["next_trade_date"] = df["trade_date"].map(next_map)
        needed_dates = set(df["trade_date"].dropna().tolist()) | set(df["next_trade_date"].dropna().tolist())
        price_df = _load_prices_daily(tushare_root, needed_dates)
        df = _fill_stock_returns(df, price_df)
        out_path = path.with_name(path.stem + "_filled.csv")
        df.to_csv(out_path, index=False)
        logger.info("写入股票收益: %s", out_path)

    for path in industry_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["trade_date"] = df["trade_date"].astype(str)
        df["next_trade_date"] = df["trade_date"].map(next_map)
        needed_dates = set(df["trade_date"].dropna().tolist()) | set(df["next_trade_date"].dropna().tolist())
        price_df = _load_prices_sw_industry(tushare_root, needed_dates)
        df = _fill_industry_returns(df, price_df)
        out_path = path.with_name(path.stem + "_filled.csv")
        df.to_csv(out_path, index=False)
        logger.info("写入行业收益: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="归因输入收益回填")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--horizon-days", type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else get_data_path("signals", "attribution", ensure=True)
    fill_returns(input_dir, args.horizon_days)


if __name__ == "__main__":
    main()
