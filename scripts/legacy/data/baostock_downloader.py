#!/usr/bin/env python3
"""
Baostock 数据下载器（A股历史/最近、申万行业）
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import baostock as bs
import pandas as pd

from scripts.data._shared.runtime import get_project_root, setup_logger

logger = setup_logger(Path(__file__).stem)


@dataclass
class BaostockConfig:
    start_date: str
    end_date: str
    output_dir: Path
    sleep_seconds: float
    max_retries: int
    backoff_factor: float


def _to_date_str(date_str: str) -> str:
    return date_str


def load_config(output_dir: Optional[str] = None) -> BaostockConfig:
    project_root = get_project_root()
    base_output = Path(output_dir) if output_dir else project_root / "data" / "baostock"
    if not base_output.is_absolute():
        base_output = project_root / base_output

    return BaostockConfig(
        start_date="2020-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        output_dir=base_output,
        sleep_seconds=0.1,
        max_retries=3,
        backoff_factor=2.0,
    )


def request_with_retry(func, cfg: BaostockConfig):
    last_error = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt >= cfg.max_retries:
                break
            wait_time = cfg.sleep_seconds * (cfg.backoff_factor ** (attempt - 1))
            logger.warning(f"请求失败，{wait_time:.2f}s 后重试: {exc}")
            time.sleep(wait_time)
    raise RuntimeError(f"请求失败，已达最大重试次数: {last_error}")


class BaostockSession:
    def __enter__(self):
        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"Baostock登录失败: {lg.error_msg}")
        logger.info("Baostock登录成功")
        return self

    def __exit__(self, exc_type, exc, tb):
        bs.logout()
        logger.info("Baostock已登出")


def get_all_stocks(query_date: str, cfg: BaostockConfig) -> pd.DataFrame:
    def _query():
        rs = bs.query_all_stock(day=query_date)
        rows = []
        while (rs.error_code == "0") & rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            raise RuntimeError(rs.error_msg)
        return pd.DataFrame(rows, columns=rs.fields)

    df = request_with_retry(_query, cfg)
    df = df[df["code"].str.match(r"sh\.6[0-9]{5}|sz\.[0-3][0-9]{5}")]
    return df[["code", "code_name", "ipoDate"]]


def download_stock_history(stock_code: str, start_date: str, end_date: str, cfg: BaostockConfig) -> pd.DataFrame:
    def _query():
        rs = bs.query_history_k_data_plus(
            stock_code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        rows = []
        while (rs.error_code == "0") & rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            raise RuntimeError(rs.error_msg)
        return pd.DataFrame(rows, columns=rs.fields)

    return request_with_retry(_query, cfg)


def download_all_history(cfg: BaostockConfig) -> Path:
    output_dir = cfg.output_dir / "history"
    output_dir.mkdir(parents=True, exist_ok=True)
    query_date = cfg.end_date

    logger.info(f"下载历史数据: {cfg.start_date} ~ {cfg.end_date}")
    stocks_df = get_all_stocks(query_date=query_date, cfg=cfg)
    logger.info(f"股票数量: {len(stocks_df)}")

    for idx, row in stocks_df.iterrows():
        stock_code = row["code"]
        stock_name = row["code_name"]
        logger.info(f"[{idx+1}/{len(stocks_df)}] {stock_code} {stock_name}")
        df = download_stock_history(stock_code, cfg.start_date, cfg.end_date, cfg)
        if not df.empty:
            df.to_parquet(output_dir / f"{stock_code}.parquet", index=False)
        time.sleep(cfg.sleep_seconds)

    return output_dir


def download_recent(cfg: BaostockConfig, start_date: str = "2024-02-06") -> Path:
    output_dir = cfg.output_dir / "recent"
    output_dir.mkdir(parents=True, exist_ok=True)
    query_date = "2024-02-05"

    logger.info(f"下载近期数据: {start_date} ~ {cfg.end_date}")
    stocks_df = get_all_stocks(query_date=query_date, cfg=cfg)
    logger.info(f"股票数量: {len(stocks_df)}")

    for idx, row in stocks_df.iterrows():
        stock_code = row["code"]
        stock_name = row["code_name"]
        logger.info(f"[{idx+1}/{len(stocks_df)}] {stock_code} {stock_name}")
        df = download_stock_history(stock_code, start_date, cfg.end_date, cfg)
        if not df.empty:
            df.to_parquet(output_dir / f"{stock_code}.parquet", index=False)
        time.sleep(cfg.sleep_seconds)

    return output_dir


def download_sw_industry(cfg: BaostockConfig) -> Path:
    output_dir = cfg.output_dir / "sw_industry"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _query_industry():
        rs = bs.query_stock_industry()
        rows = []
        while (rs.error_code == "0") & rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code != "0":
            raise RuntimeError(rs.error_msg)
        return pd.DataFrame(rows, columns=rs.fields)

    df_industry = request_with_retry(_query_industry, cfg)
    df_industry.to_parquet(output_dir / "sw_industry_list.parquet", index=False)

    sw_l1 = df_industry[df_industry["level"] == "1.0"]
    for idx, row in sw_l1.iterrows():
        industry_code = row["industry"]
        industry_name = row["industry_name"]
        logger.info(f"[{idx+1}/{len(sw_l1)}] {industry_name} {industry_code}")

        df = download_stock_history(industry_code, cfg.start_date, cfg.end_date, cfg)
        if not df.empty:
            df.to_parquet(output_dir / f"{industry_code}.parquet", index=False)
        time.sleep(cfg.sleep_seconds)

    return output_dir


def test_one_stock(output_dir: Optional[str] = None):
    cfg = load_config(output_dir=output_dir)
    cfg.start_date = "2024-01-01"
    cfg.end_date = "2024-01-31"
    with BaostockSession():
        df = download_stock_history("sh.600000", cfg.start_date, cfg.end_date, cfg)
    logger.info(f"测试样例条数: {len(df)}")


def test_sw_industry_one(output_dir: Optional[str] = None):
    cfg = load_config(output_dir=output_dir)
    cfg.start_date = "2024-01-01"
    cfg.end_date = "2024-01-31"
    with BaostockSession():
        download_sw_industry(cfg)


def main():
    # with BaostockSession():
    #     cfg = load_config(output_dir="/tmp/sage_data")
    #     download_all_history(cfg)
    #
    # with BaostockSession():
    #     cfg = load_config(output_dir="/tmp/sage_data")
    #     download_recent(cfg, start_date="2024-02-06")
    #
    # with BaostockSession():
    #     cfg = load_config(output_dir="/tmp/sage_data")
    #     download_sw_industry(cfg)
    #
    # test_one_stock(output_dir="/tmp/sage_data")
    # test_sw_industry_one(output_dir="/tmp/sage_data")
    pass


if __name__ == "__main__":
    main()
