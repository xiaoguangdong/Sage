#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tushare 数据工具集合（统一入口）

说明：
1) 所有日期参数默认使用 YYYYMMDD
2) 输出目录可通过参数指定（默认读取 config/base.yaml -> data.download.output_dir）
3) main() 中提供所有功能的调用示例（已注释）
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import argparse
import tushare as ts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_project_root, get_tushare_token, setup_logger, get_data_root, get_data_path

logger = setup_logger(Path(__file__).stem)


@dataclass
class DownloadConfig:
    output_dir: Path
    sleep_seconds: float
    max_retries: int
    backoff_factor: float
    page_limit_default: int
    states_dir: Path

    @property
    def state_dir(self) -> Path:
        return self.states_dir


def _to_yyyymmdd(date_str: str) -> str:
    value = date_str.replace("-", "")
    if len(value) != 8:
        raise ValueError(f"日期格式错误，需 YYYYMMDD: {date_str}")
    return value


def _resolve_output_root(output_dir: Optional[str]) -> Path:
    cfg = load_config()
    root = Path(output_dir) if output_dir else cfg.output_dir
    if not root.is_absolute():
        root = get_data_root("primary") / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tushare_root(output_dir: Optional[str]) -> Path:
    root = _resolve_output_root(output_dir)
    path = root / "tushare"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    output_dir = download_cfg.get("output_dir", "raw")
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = get_data_root("primary") / output_dir

    cfg = DownloadConfig(
        output_dir=output_dir,
        sleep_seconds=float(download_cfg.get("sleep_seconds", 60)),
        max_retries=int(download_cfg.get("max_retries", 3)),
        backoff_factor=float(download_cfg.get("backoff_factor", 2.0)),
        page_limit_default=int(download_cfg.get("page_limit_default", 4000)),
        states_dir=get_data_path("states", ensure=True),
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
    def __init__(self, pro, cfg: DownloadConfig, name: str, limit: int, output_root: Path):
        self.pro = pro
        self.cfg = cfg
        self.name = name
        self.limit = limit
        self.output_root = output_root
        self.state_path = cfg.state_dir / f"{name}.json"
        self.parts_dir = output_root / name / "parts"
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
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    resume: bool = True,
    limit: Optional[int] = None,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    tushare_root = _tushare_root(output_dir)
    pro = ts.pro_api(get_tushare_token())
    downloader = PagedDownloader(pro.daily_basic, cfg, "daily_basic", limit or 6000, tushare_root)
    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)
    return downloader.download(lambda offset, page_limit: {
        "start_date": start,
        "end_date": end,
        "offset": offset,
        "limit": page_limit,
    }, resume=resume, meta={"start_date": start, "end_date": end, "limit": downloader.limit})


def download_margin(
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    resume: bool = True,
    limit: Optional[int] = None,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    tushare_root = _tushare_root(output_dir)
    pro = ts.pro_api(get_tushare_token())
    downloader = PagedDownloader(pro.margin, cfg, "margin", limit or 4000, tushare_root)
    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)
    return downloader.download(lambda offset, page_limit: {
        "start_date": start,
        "end_date": end,
        "offset": offset,
        "limit": page_limit,
    }, resume=resume, meta={"start_date": start, "end_date": end, "limit": downloader.limit})


def download_daily_kline(
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    resume: bool = True,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    tushare_root = _tushare_root(output_dir)
    output_path = tushare_root / "daily"
    output_path.mkdir(parents=True, exist_ok=True)
    state_file = output_path / "download_state.json"

    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)

    pro = ts.pro_api(get_tushare_token())

    completed = set()
    if resume and state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            completed = set(state.get("completed_dates", []))
        except Exception:
            completed = set()

    trade_cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end)
    trade_dates = trade_cal[trade_cal["is_open"] == 1]["cal_date"].tolist()

    year_frames: List[pd.DataFrame] = []
    current_year = None

    def flush_year(year: str, frames: List[pd.DataFrame]):
        if not frames:
            return
        df_year = pd.concat(frames, ignore_index=True)
        file_path = output_path / f"daily_{year}.parquet"
        if file_path.exists():
            existing = pd.read_parquet(file_path)
            df_year = pd.concat([existing, df_year], ignore_index=True)
        df_year = df_year.sort_values("trade_date")
        df_year.to_parquet(file_path, index=False)

    for td in trade_dates:
        if td in completed:
            continue
        df = request_with_retry(lambda: pro.daily(trade_date=td), cfg)
        if df is None or df.empty:
            completed.add(td)
            continue
        year = td[:4]
        if current_year and year != current_year:
            flush_year(current_year, year_frames)
            year_frames = []
        year_frames.append(df)
        current_year = year
        completed.add(td)
        state_file.write_text(json.dumps({"completed_dates": sorted(completed)}, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(cfg.sleep_seconds)

    if current_year:
        flush_year(current_year, year_frames)

    return output_path


def download_index_ohlc(
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    indices: Optional[List[Tuple[str, str]]] = None,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    tushare_root = _tushare_root(output_dir)
    output_path = tushare_root / "index"
    output_path.mkdir(parents=True, exist_ok=True)

    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)
    pro = ts.pro_api(get_tushare_token())

    if not indices:
        indices = [("000300.SH", "沪深300"), ("000905.SH", "中证500")]

    all_frames = []
    for code, name in indices:
        df = request_with_retry(lambda: pro.daily(ts_code=code, start_date=start, end_date=end), cfg)
        if df is None or df.empty:
            logger.warning(f"{code} 无数据")
            continue
        df = df.rename(columns={"trade_date": "date", "ts_code": "code", "pct_chg": "pct_change"})
        df.to_parquet(output_path / f"index_{code.replace('.', '_')}_ohlc.parquet", index=False)
        all_frames.append(df)
        time.sleep(cfg.sleep_seconds)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_parquet(output_path / "index_ohlc_all.parquet", index=False)

    return output_path


def download_hs300_constituents(
    start_year: int,
    end_year: int,
    output_dir: Optional[str] = None,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    tushare_root = _tushare_root(output_dir)
    output_path = tushare_root / "constituents"
    output_path.mkdir(parents=True, exist_ok=True)

    pro = ts.pro_api(get_tushare_token())
    all_frames = []
    for year in range(start_year, end_year + 1):
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        df = request_with_retry(lambda: pro.index_weight(index_code="000300.SH", start_date=start_date, end_date=end_date), cfg)
        if df is None or df.empty:
            continue
        df.to_parquet(output_path / f"hs300_constituents_{year}.parquet", index=False)
        all_frames.append(df)
        time.sleep(cfg.sleep_seconds)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_parquet(output_path / "hs300_constituents_all.parquet", index=False)

    return output_path


def download_hs300_moneyflow(
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    tushare_root = _tushare_root(output_dir)
    constituents_path = tushare_root / "constituents" / "hs300_constituents_all.parquet"
    if not constituents_path.exists():
        raise FileNotFoundError(f"成分股文件不存在: {constituents_path}")

    pro = ts.pro_api(get_tushare_token())
    df_constituents = pd.read_parquet(constituents_path)
    stocks = df_constituents["con_code"].unique().tolist()

    output_path = tushare_root / "moneyflow"
    output_path.mkdir(parents=True, exist_ok=True)

    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)

    all_frames = []
    for code in stocks:
        df = request_with_retry(lambda: pro.moneyflow(ts_code=code, start_date=start, end_date=end), cfg)
        if df is not None and not df.empty:
            all_frames.append(df)
        time.sleep(cfg.sleep_seconds)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_parquet(output_path / "moneyflow_all.parquet", index=False)
        for code in stocks:
            part = combined[combined["ts_code"] == code]
            if not part.empty:
                part.to_parquet(output_path / f"{code}.parquet", index=False)

    return output_path


def download_sw_industry_classify(output_dir: Optional[str] = None) -> Path:
    tushare_root = _tushare_root(output_dir)
    output_path = tushare_root / "sw_industry"
    output_path.mkdir(parents=True, exist_ok=True)

    pro = ts.pro_api(get_tushare_token())
    df = pro.index_classify(level="L1", src="SW2021")
    if df is not None and not df.empty:
        df.to_parquet(output_path / "sw_industry_l1.parquet", index=False)
    return output_path


def download_sw_industry_daily(
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    tushare_root = _tushare_root(output_dir)
    output_path = tushare_root / "sw_industry"
    output_path.mkdir(parents=True, exist_ok=True)
    l1_path = output_path / "sw_industry_l1.parquet"
    if not l1_path.exists():
        raise FileNotFoundError(f"行业分类文件不存在: {l1_path}")

    df_industry = pd.read_parquet(l1_path)
    pro = ts.pro_api(get_tushare_token())
    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)

    all_frames = []
    for _, row in df_industry.iterrows():
        code = row["index_code"]
        name = row["industry_name"]
        df = request_with_retry(lambda: pro.daily(ts_code=code, start_date=start, end_date=end), cfg)
        if df is not None and not df.empty:
            df["industry_name"] = name
            all_frames.append(df)
        time.sleep(cfg.sleep_seconds)

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_parquet(output_path / "sw_industry_daily.parquet", index=False)

    return output_path


def download_opt_daily(
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    exchanges: Optional[List[str]] = None,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    tushare_root = _tushare_root(output_dir)
    output_path = tushare_root / "options"
    output_path.mkdir(parents=True, exist_ok=True)

    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)
    pro = ts.pro_api(get_tushare_token())

    dates = pd.date_range(datetime.strptime(start, "%Y%m%d"), datetime.strptime(end, "%Y%m%d")).strftime("%Y%m%d").tolist()
    exchanges = exchanges or ["SSE", "SZSE"]

    for exchange in exchanges:
        frames = []
        for date in dates:
            df = request_with_retry(lambda: pro.opt_daily(trade_date=date, exchange=exchange, calender="natural"), cfg)
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(cfg.sleep_seconds)
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(output_path / f"opt_daily_{exchange.lower()}.parquet", index=False)

    return output_path


def download_fina_indicator(
    stock_list_csv: str,
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    tushare_root = _tushare_root(output_dir)
    output_path = tushare_root / "fina_indicator"
    output_path.mkdir(parents=True, exist_ok=True)

    pro = ts.pro_api(get_tushare_token())
    stocks = pd.read_csv(stock_list_csv)
    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)

    for _, row in stocks.iterrows():
        ts_code = row["tushare"] if "tushare" in row else row.get("ts_code")
        if not ts_code:
            continue
        df = request_with_retry(lambda: pro.fina_indicator(ts_code=ts_code, start_date=start, end_date=end), cfg)
        if df is not None and not df.empty:
            df.to_parquet(output_path / f"fina_indicator_{ts_code.replace('.', '_')}_{start}_{end}.parquet", index=False)
        time.sleep(cfg.sleep_seconds)

    return output_path


def download_fina_indicator_vip(
    start_year: int,
    end_year: int,
    output_dir: Optional[str] = None,
    sleep_seconds: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds

    tushare_root = _tushare_root(output_dir)
    output_path = tushare_root / "fina_indicator_vip"
    output_path.mkdir(parents=True, exist_ok=True)

    pro = ts.pro_api(get_tushare_token())

    for year in range(start_year, end_year + 1):
        for quarter in ["0331", "0630", "0930", "1231"]:
            period = f"{year}{quarter}"
            df = request_with_retry(lambda: pro.fina_indicator_vip(period=period), cfg)
            if df is not None and not df.empty:
                df.to_parquet(output_path / f"fina_indicator_vip_{period}.parquet", index=False)
            time.sleep(cfg.sleep_seconds)

    return output_path


def fetch_tushare_sectors(output_dir: Optional[str] = None) -> Path:
    tushare_root = _tushare_root(output_dir)
    output_path = tushare_root / "sectors"
    output_path.mkdir(parents=True, exist_ok=True)

    pro = ts.pro_api(get_tushare_token())

    df = pro.index_classify(level="L1", source="SW")
    if df is not None and not df.empty:
        df.to_csv(output_path / "SW_L1_classify.csv", index=False, encoding="utf-8-sig")

    concepts = pro.concept()
    if concepts is not None and not concepts.empty:
        concepts.to_csv(output_path / "concepts.csv", index=False, encoding="utf-8-sig")

    return output_path


class ConceptDataManager:
    def __init__(self, data_dir: Path, timeout: int = 30):
        self.pro = ts.pro_api(get_tushare_token())
        self.timeout = timeout
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_concept_list(self):
        concepts = self.pro.concept()
        logger.info(f"获取概念数: {len(concepts)}")
        return concepts

    def fetch_concept_members(self, concept_id, max_retries=3):
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                members = self.pro.concept_detail(id=concept_id)
                if time.time() - start_time > self.timeout:
                    return None
                return members
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    logger.error(f"概念 {concept_id} 获取失败: {e}")
                    return None
        return None

    def fetch_all_concept_members(self, concepts=None):
        if concepts is None:
            concepts = self.fetch_concept_list()
        all_members = []
        for _, row in concepts.iterrows():
            members = self.fetch_concept_members(row["code"])
            if members is not None and len(members) > 0:
                members["concept_id"] = row["code"]
                members["concept_name"] = row["name"]
                members["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
                all_members.append(members)
            time.sleep(0.3)
        if all_members:
            return pd.concat(all_members, ignore_index=True)
        return None

    def save_concept_data(self, data: pd.DataFrame, suffix: str = "") -> Path:
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"all_concept_details_{timestamp}{suffix}.csv"
        filepath = self.data_dir / filename
        data.to_csv(filepath, index=False, encoding="utf-8-sig")
        latest_path = self.data_dir / f"all_concept_details{suffix}.csv"
        data.to_csv(latest_path, index=False, encoding="utf-8-sig")
        return filepath

    def load_concept_data(self, filename="all_concept_details.csv"):
        filepath = self.data_dir / filename
        if not filepath.exists():
            return None
        return pd.read_csv(filepath)

    def load_stock_data(self, start_date, end_date, daily_dir: Path):
        year = start_date[:4]
        filepath = daily_dir / f"daily_{year}.parquet"
        if not filepath.exists():
            return None
        daily_data = pd.read_parquet(filepath)
        daily_data["trade_date"] = pd.to_datetime(daily_data["trade_date"])
        period_data = daily_data[
            (daily_data["trade_date"] >= pd.Timestamp(start_date)) &
            (daily_data["trade_date"] <= pd.Timestamp(end_date))
        ]
        return period_data

    def calculate_concept_performance(self, concept_data, stock_data, min_stock_count=10):
        stock_count = concept_data.groupby("concept_name")["ts_code"].count()
        valid_concepts = stock_count[stock_count >= min_stock_count].index.tolist()
        concept_filtered = concept_data[concept_data["concept_name"].isin(valid_concepts)]

        concept_performance = []
        stock_data["week_start"] = stock_data["trade_date"] - pd.to_timedelta(stock_data["trade_date"].dt.dayofweek, unit="D")
        stock_data["year_week"] = stock_data["week_start"].dt.strftime("%Y-%U")

        for concept_name in concept_filtered["concept_name"].unique():
            stocks = concept_filtered[concept_filtered["concept_name"] == concept_name]["ts_code"].tolist()
            concept_stocks = stock_data[stock_data["ts_code"].isin(stocks)]
            if len(concept_stocks) == 0:
                continue
            concept_daily = concept_stocks.groupby("trade_date")["pct_chg"].mean().reset_index()
            concept_daily.columns = ["trade_date", "concept_return"]
            concept_cumulative = (1 + concept_daily["concept_return"] / 100).cumprod() - 1
            total_return = concept_cumulative.iloc[-1] * 100
            max_drawdown = (concept_cumulative.cummax() - concept_cumulative).max() * 100
            volatility = concept_daily["concept_return"].std()
            concept_performance.append({
                "concept_name": concept_name,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "volatility": volatility,
                "stock_count": len(stocks),
            })

        return pd.DataFrame(concept_performance)


def concept_update_tushare(
    mode: str,
    start_date: str,
    end_date: str,
    min_stock_count: int = 10,
    output_dir: Optional[str] = None,
) -> Path:
    tushare_root = _tushare_root(output_dir)
    sectors_dir = tushare_root / "sectors"
    daily_dir = tushare_root / "daily"
    manager = ConceptDataManager(data_dir=sectors_dir)

    if mode == "init":
        data = manager.fetch_all_concept_members()
        if data is not None:
            manager.save_concept_data(data, suffix="_base")
        return sectors_dir

    if mode in ("update", "calculate"):
        concept_data = manager.load_concept_data("all_concept_details_base.csv")
        if concept_data is None:
            raise FileNotFoundError("缺少基准概念数据 all_concept_details_base.csv")
        stock_data = manager.load_stock_data(start_date, end_date, daily_dir)
        if stock_data is None:
            raise FileNotFoundError("缺少个股日线数据")
        performance = manager.calculate_concept_performance(concept_data, stock_data, min_stock_count=min_stock_count)
        if performance is not None:
            ts = datetime.now().strftime("%Y%m%d")
            performance.to_csv(sectors_dir / f"concept_performance_{ts}.csv", index=False, encoding="utf-8-sig")
        if mode == "update":
            new_data = manager.fetch_all_concept_members()
            if new_data is not None:
                manager.save_concept_data(new_data)
        return sectors_dir

    raise ValueError("mode 需为 init/update/calculate")


class EastMoneyConceptManager:
    def __init__(self, data_dir: Path, timeout: int = 30):
        self.pro = ts.pro_api(get_tushare_token())
        self.timeout = timeout
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.market = "EB"

    def fetch_concept_daily(self, trade_date=None, max_retries=3):
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                if trade_date:
                    data = self.pro.dc_index(market=self.market, trade_date=trade_date)
                else:
                    data = self.pro.dc_index(market=self.market)
                if time.time() - start_time > self.timeout:
                    return None
                return data
            except Exception:
                time.sleep(1)
        return None

    def fetch_concept_members(self, trade_date=None, max_retries=3):
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                if trade_date:
                    data = self.pro.dc_member(market=self.market, trade_date=trade_date)
                else:
                    data = self.pro.dc_member(market=self.market)
                if time.time() - start_time > self.timeout:
                    return None
                return data
            except Exception:
                time.sleep(1)
        return None


def concept_update_eastmoney(
    mode: str,
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
) -> Path:
    tushare_root = _tushare_root(output_dir)
    sectors_dir = tushare_root / "sectors"
    manager = EastMoneyConceptManager(data_dir=sectors_dir)

    if mode == "init":
        data = manager.fetch_concept_daily()
        if data is not None:
            data.to_csv(sectors_dir / "concepts_eastmoney_base.csv", index=False, encoding="utf-8-sig")
        return sectors_dir

    if mode in ("update", "calculate"):
        daily = manager.fetch_concept_daily()
        members = manager.fetch_concept_members()
        if daily is not None:
            daily.to_csv(sectors_dir / f"concepts_eastmoney_{datetime.now().strftime('%Y%m%d')}.csv", index=False, encoding="utf-8-sig")
        if members is not None:
            members.to_csv(sectors_dir / f"concept_members_eastmoney_{datetime.now().strftime('%Y%m%d')}.csv", index=False, encoding="utf-8-sig")
        return sectors_dir

    raise ValueError("mode 需为 init/update/calculate")


def fetch_hybrid_concepts(output_dir: Optional[str] = None):
    try:
        import efinance as ef  # type: ignore
    except Exception as exc:
        raise RuntimeError("需要安装 efinance") from exc

    tushare_root = _tushare_root(output_dir)
    output_path = tushare_root / "concepts_hybrid"
    output_path.mkdir(parents=True, exist_ok=True)

    pro = ts.pro_api(get_tushare_token())
    concepts = pro.concept()
    concepts.to_csv(output_path / "tushare_concepts.csv", index=False, encoding="utf-8-sig")

    return output_path


def main():
    # download_daily_basic(start_date="20240101", end_date="20240105", output_dir="/tmp/sage_data", resume=False)
    # download_margin(start_date="20240101", end_date="20240105", output_dir="/tmp/sage_data", resume=False)
    # download_daily_kline(start_date="20240101", end_date="20240105", output_dir="/tmp/sage_data", resume=False)
    # download_index_ohlc(start_date="20240101", end_date="20240105", output_dir="/tmp/sage_data")
    # download_hs300_constituents(start_year=2020, end_year=2021, output_dir="/tmp/sage_data")
    # download_hs300_moneyflow(start_date="20240101", end_date="20240105", output_dir="/tmp/sage_data")
    # download_sw_industry_classify(output_dir="/tmp/sage_data")
    # download_sw_industry_daily(start_date="20240101", end_date="20240105", output_dir="/tmp/sage_data")
    # download_opt_daily(start_date="20240101", end_date="20240105", output_dir="/tmp/sage_data")
    # download_fina_indicator(stock_list_csv="data/raw/tushare/filtered_stocks_list.csv", start_date="20240101", end_date="20241231", output_dir="/tmp/sage_data")
    # download_fina_indicator_vip(start_year=2020, end_year=2021, output_dir="/tmp/sage_data")
    # fetch_tushare_sectors(output_dir="/tmp/sage_data")
    # concept_update_tushare(mode="init", start_date="20240924", end_date="20241231", output_dir="/tmp/sage_data")
    # concept_update_tushare(mode="update", start_date="20240924", end_date="20241231", output_dir="/tmp/sage_data")
    # concept_update_eastmoney(mode="init", start_date="20240924", end_date="20241231", output_dir="/tmp/sage_data")
    parser = argparse.ArgumentParser(description="Tushare 数据工具集合")
    parser.add_argument("--action", type=str, default=None)
    parser.add_argument("--start-date", type=str, default="20240101")
    parser.add_argument("--end-date", type=str, default="20240105")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=None)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--stock-list-csv", type=str, default=str(get_data_path("raw", "tushare", "filtered_stocks_list.csv")))
    parser.add_argument("--mode", type=str, default="update")
    parser.add_argument("--min-stock-count", type=int, default=10)
    args = parser.parse_args()

    if not args.action:
        return

    if args.action == "daily_basic":
        download_daily_basic(args.start_date, args.end_date, args.output_dir, args.resume, args.limit, args.sleep_seconds)
        return
    if args.action == "margin":
        download_margin(args.start_date, args.end_date, args.output_dir, args.resume, args.limit, args.sleep_seconds)
        return
    if args.action == "daily_kline":
        download_daily_kline(args.start_date, args.end_date, args.output_dir, args.resume, args.sleep_seconds)
        return
    if args.action == "index_ohlc":
        download_index_ohlc(args.start_date, args.end_date, args.output_dir, sleep_seconds=args.sleep_seconds)
        return
    if args.action == "hs300_constituents":
        download_hs300_constituents(args.start_year, args.end_year, args.output_dir, args.sleep_seconds)
        return
    if args.action == "hs300_moneyflow":
        download_hs300_moneyflow(args.start_date, args.end_date, args.output_dir, args.sleep_seconds)
        return
    if args.action == "sw_industry_classify":
        download_sw_industry_classify(args.output_dir)
        return
    if args.action == "sw_industry_daily":
        download_sw_industry_daily(args.start_date, args.end_date, args.output_dir, args.sleep_seconds)
        return
    if args.action == "opt_daily":
        download_opt_daily(args.start_date, args.end_date, args.output_dir, sleep_seconds=args.sleep_seconds)
        return
    if args.action == "fina_indicator":
        download_fina_indicator(args.stock_list_csv, args.start_date, args.end_date, args.output_dir, args.sleep_seconds)
        return
    if args.action == "fina_indicator_vip":
        download_fina_indicator_vip(args.start_year, args.end_year, args.output_dir, args.sleep_seconds)
        return
    if args.action == "tushare_sectors":
        fetch_tushare_sectors(args.output_dir)
        return
    if args.action == "concept_update_tushare":
        concept_update_tushare(args.mode, args.start_date, args.end_date, args.min_stock_count, args.output_dir)
        return
    if args.action == "concept_update_eastmoney":
        concept_update_eastmoney(args.mode, args.start_date, args.end_date, args.output_dir)
        return


if __name__ == "__main__":
    main()
