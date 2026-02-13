#!/usr/bin/env python3
"""
Tushare 数据下载器（暂支持 daily_basic 与 margin）
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import tushare as ts

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_project_root, get_tushare_token, setup_logger

logger = setup_logger(Path(__file__).stem)


@dataclass
class DownloadConfig:
    output_dir: Path
    sleep_seconds: float
    max_retries: int
    backoff_factor: float
    page_limit_default: int

    @property
    def state_dir(self) -> Path:
        return self.output_dir / "states"


def _to_ts_date(date_str: str) -> str:
    if "-" in date_str:
        return date_str.replace("-", "")
    return date_str


def load_config(config_path: Optional[Path] = None) -> DownloadConfig:
    project_root = get_project_root()
    path = config_path or project_root / "config" / "base.yaml"
    config = {}
    if path.exists():
        try:
            import yaml  # type: ignore
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning(f"读取配置失败，使用默认值: {exc}")

    download_cfg = (config.get("data") or {}).get("download") or {}

    output_dir = download_cfg.get("output_dir", "/tmp/sage_data")
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    cfg = DownloadConfig(
        output_dir=output_dir,
        sleep_seconds=float(download_cfg.get("sleep_seconds", 60)),
        max_retries=int(download_cfg.get("max_retries", 3)),
        backoff_factor=float(download_cfg.get("backoff_factor", 2.0)),
        page_limit_default=int(download_cfg.get("page_limit_default", 4000)),
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def request_with_retry(func: Callable[[], pd.DataFrame], cfg: DownloadConfig) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt >= cfg.max_retries:
                break
            wait_time = cfg.sleep_seconds * (cfg.backoff_factor ** (attempt - 1))
            logger.warning(f"请求失败，{wait_time:.1f}s 后重试: {exc}")
            time.sleep(wait_time)
    raise RuntimeError(f"请求失败，已达最大重试次数: {last_error}")


class PagedDownloader:
    def __init__(self, pro, cfg: DownloadConfig, name: str, limit: int):
        self.pro = pro
        self.cfg = cfg
        self.name = name
        self.limit = limit
        self.state_path = cfg.state_dir / f"{name}.json"
        self.parts_dir = cfg.output_dir / name / "parts"
        self.parts_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self, resume: bool, meta: Optional[dict]) -> int:
        if resume and self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                if meta and state.get("meta") != meta:
                    return 0
                return int(state.get("offset", 0))
            except Exception as exc:
                logger.warning(f"读取断点状态失败，重置 offset: {exc}")
        return 0

    def save_state(self, offset: int, meta: Optional[dict]):
        state = {
            "offset": offset,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "meta": meta or {},
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def download(self, params_builder: Callable[[int, int], dict], resume: bool = True, meta: Optional[dict] = None) -> Path:
        offset = self.load_state(resume, meta)
        logger.info(f"{self.name} 下载开始，offset={offset}")

        while True:
            params = params_builder(offset, self.limit)
            logger.info(f"{self.name} 请求 offset={offset}, limit={self.limit}")

            df = request_with_retry(lambda: self.pro(**params), self.cfg)
            if df is None or df.empty:
                logger.info(f"{self.name} 无数据返回，停止")
                break

            part_path = self.parts_dir / f"part_{offset:010d}.parquet"
            df.to_parquet(part_path, index=False)
            logger.info(f"{self.name} 保存分片: {part_path} ({len(df)} 条)")

            offset += len(df)
            self.save_state(offset, meta)

            if len(df) < self.limit:
                logger.info(f"{self.name} 已到最后一页")
                break

            time.sleep(self.cfg.sleep_seconds)

        return self.parts_dir


def download_daily_basic(
    token: str,
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    resume: bool = True,
    limit: Optional[int] = None,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if output_dir:
        cfg.output_dir = Path(output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    pro = ts.pro_api(token)
    downloader = PagedDownloader(pro.daily_basic, cfg, "daily_basic", limit or 6000)
    start = _to_ts_date(start_date)
    end = _to_ts_date(end_date)
    return downloader.download(lambda offset, page_limit: {
        "start_date": start,
        "end_date": end,
        "offset": offset,
        "limit": page_limit,
    }, resume=resume, meta={"start_date": start, "end_date": end, "limit": downloader.limit})


def download_margin(
    token: str,
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    resume: bool = True,
    limit: Optional[int] = None,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if output_dir:
        cfg.output_dir = Path(output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    pro = ts.pro_api(token)
    downloader = PagedDownloader(pro.margin, cfg, "margin", limit or 4000)
    start = _to_ts_date(start_date)
    end = _to_ts_date(end_date)
    return downloader.download(lambda offset, page_limit: {
        "start_date": start,
        "end_date": end,
        "offset": offset,
        "limit": page_limit,
    }, resume=resume, meta={"start_date": start, "end_date": end, "limit": downloader.limit})


def test_daily_basic_small(output_dir: Optional[str] = None):
    token = get_tushare_token()
    return download_daily_basic(
        token=token,
        start_date="2024-01-01",
        end_date="2024-01-31",
        output_dir=output_dir,
        resume=False,
        limit=2000,
        sleep_seconds=5,
    )


def test_margin_small(output_dir: Optional[str] = None):
    token = get_tushare_token()
    return download_margin(
        token=token,
        start_date="2024-01-01",
        end_date="2024-01-31",
        output_dir=output_dir,
        resume=False,
        limit=2000,
        sleep_seconds=5,
    )


def main():
    # download_daily_basic(get_tushare_token(), "2020-01-01", "2024-12-31", output_dir="/tmp/sage_data")
    # download_margin(get_tushare_token(), "2020-01-01", "2024-12-31", output_dir="/tmp/sage_data")
    # test_daily_basic_small(output_dir="/tmp/sage_data")
    # test_margin_small(output_dir="/tmp/sage_data")

    parser = argparse.ArgumentParser(description="Tushare 数据下载器")
    parser.add_argument("--endpoint", choices=["daily_basic", "margin"], help="接口名称")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=None)
    args = parser.parse_args()

    if not args.endpoint:
        logger.info("未指定接口，直接退出")
        return

    token = get_tushare_token()

    if args.endpoint == "daily_basic":
        download_daily_basic(
            token=token,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            resume=args.resume,
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
        )
        return

    if args.endpoint == "margin":
        download_margin(
            token=token,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            resume=args.resume,
            limit=args.limit,
            sleep_seconds=args.sleep_seconds,
        )
        return


if __name__ == "__main__":
    main()
