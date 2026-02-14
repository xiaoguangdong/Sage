#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tushare 政策文本拉取（公告/研报）

支持：
1) 上市公司全量公告（anns_d）
2) 券商研究报告（research_report）

输出：
- data/raw/tushare/policy/tushare_anns.parquet
- data/raw/tushare/policy/tushare_reports.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import tushare as ts

from scripts.data._shared.runtime import disable_proxy, get_data_path, get_tushare_token, setup_logger


logger = setup_logger(Path(__file__).stem, module="data")


@dataclass
class DownloadConfig:
    output_dir: Path
    sleep_seconds: float = 60.0
    max_retries: int = 3
    backoff_factor: float = 2.0
    page_limit: int = 2000
    state_dir: Path = get_data_path("states", ensure=True)


def request_with_retry(func: Callable[[], pd.DataFrame], cfg: DownloadConfig) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            wait_time = cfg.sleep_seconds * (cfg.backoff_factor ** (attempt - 1))
            logger.warning(f"请求失败，{wait_time:.1f}s 后重试: {exc}")
            time.sleep(wait_time)
    raise RuntimeError(f"请求失败，已达最大重试次数: {last_error}")


def _load_state(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("offset", 0))
    except Exception:
        return 0


def _save_state(path: Path, offset: int, meta: Dict):
    payload = {
        "offset": offset,
        "meta": meta,
        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_and_save(path: Path, df: pd.DataFrame, subset: list):
    if path.exists():
        old = pd.read_parquet(path)
        combined = pd.concat([old, df], ignore_index=True)
    else:
        combined = df
    combined = combined.drop_duplicates(subset=subset)
    combined.to_parquet(path, index=False)


def fetch_anns(pro, cfg: DownloadConfig, start_date: str, end_date: str, resume: bool = True) -> Path:
    output_path = cfg.output_dir / "tushare_anns.parquet"
    state_path = cfg.state_dir / "tushare_anns.json"
    offset = _load_state(state_path) if resume else 0

    logger.info(f"开始拉取公告 anns_d，offset={offset}")
    while True:
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "offset": offset,
            "limit": cfg.page_limit,
        }
        df = request_with_retry(lambda: pro.anns_d(**params), cfg)
        if df is None or df.empty:
            break
        _merge_and_save(output_path, df, subset=["ann_date", "ts_code", "title", "url"])
        offset += len(df)
        _save_state(state_path, offset, params)
        if len(df) < cfg.page_limit:
            break
        time.sleep(cfg.sleep_seconds)
    logger.info(f"公告保存完成: {output_path}")
    return output_path


def fetch_reports(pro, cfg: DownloadConfig, start_date: str, end_date: str, resume: bool = True) -> Path:
    output_path = cfg.output_dir / "tushare_reports.parquet"
    state_path = cfg.state_dir / "tushare_reports.json"
    offset = _load_state(state_path) if resume else 0

    logger.info(f"开始拉取研报 research_report，offset={offset}")
    while True:
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "offset": offset,
            "limit": min(cfg.page_limit, 1000),
        }
        df = request_with_retry(lambda: pro.research_report(**params), cfg)
        if df is None or df.empty:
            break
        _merge_and_save(output_path, df, subset=["trade_date", "title", "ts_code", "inst_csname"])
        offset += len(df)
        _save_state(state_path, offset, params)
        if len(df) < params["limit"]:
            break
        time.sleep(cfg.sleep_seconds)
    logger.info(f"研报保存完成: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True, help="开始日期YYYYMMDD")
    parser.add_argument("--end-date", required=True, help="结束日期YYYYMMDD")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认 data/raw/tushare/policy）")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--action", choices=["anns", "reports", "all"], default="all")
    parser.add_argument("--sleep-seconds", type=float, default=60.0)
    parser.add_argument("--page-limit", type=int, default=2000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("raw", "tushare", "policy", ensure=True)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = DownloadConfig(
        output_dir=output_dir,
        sleep_seconds=args.sleep_seconds,
        page_limit=args.page_limit,
    )

    disable_proxy()
    pro = ts.pro_api(get_tushare_token())

    if args.action in ("anns", "all"):
        fetch_anns(pro, cfg, args.start_date, args.end_date, resume=args.resume)
    if args.action in ("reports", "all"):
        fetch_reports(pro, cfg, args.start_date, args.end_date, resume=args.resume)


if __name__ == "__main__":
    main()
