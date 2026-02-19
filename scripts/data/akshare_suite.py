#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Akshare 数据工具集合（统一入口）

说明：
1) 所有日期参数默认使用 YYYYMMDD
2) 输出目录可通过参数指定（默认读取 config/base.yaml -> data.download.output_dir）
3) main() 中提供所有功能的调用示例（已注释）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import akshare as ak
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_data_path, get_data_root, get_project_root, setup_logger

logger = setup_logger(Path(__file__).stem)


@dataclass
class DownloadConfig:
    output_dir: Path
    sleep_seconds: float
    max_retries: int
    backoff_factor: float
    states_dir: Path

    @property
    def state_dir(self) -> Path:
        return self.states_dir


def _to_yyyymmdd(date_str: str) -> str:
    value = date_str.replace("-", "")
    if len(value) != 8:
        raise ValueError(f"日期格式错误，需 YYYYMMDD: {date_str}")
    return value


def load_config(config_path: Optional[Path] = None) -> DownloadConfig:
    project_root = get_project_root()
    path = config_path or project_root / "config" / "base.yaml"
    config: Dict[str, object] = {}
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
        states_dir=get_data_path("states", ensure=True),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _resolve_output_root(output_dir: Optional[str]) -> Path:
    cfg = load_config()
    root = Path(output_dir) if output_dir else cfg.output_dir
    if not root.is_absolute():
        root = get_data_root("primary") / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def _akshare_root(output_dir: Optional[str]) -> Path:
    root = _resolve_output_root(output_dir)
    path = root / "akshare"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in value.strip())


def _load_state(path: Path) -> Dict[str, object]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"读取断点状态失败: {exc}")
    return {}


def _save_state(path: Path, payload: Dict[str, object]) -> None:
    payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def download_concept_list(
    output_dir: Optional[str] = None,
    sleep_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds
    if max_retries is not None:
        cfg.max_retries = max_retries
    if backoff_factor is not None:
        cfg.backoff_factor = backoff_factor

    ak_root = _akshare_root(output_dir)
    concept_dir = ak_root / "concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)

    df = request_with_retry(lambda: ak.stock_board_concept_name_em(), cfg)
    if df is None or df.empty:
        raise RuntimeError("未获取到概念列表数据")

    parquet_path = concept_dir / "concept_list.parquet"
    csv_path = concept_dir / "concept_list.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存概念列表: {parquet_path}")
    return parquet_path


def _extract_concept_fields(df: pd.DataFrame) -> List[Tuple[str, str]]:
    name_col = "板块名称" if "板块名称" in df.columns else "名称"
    code_col = "板块代码" if "板块代码" in df.columns else "代码"
    concepts: List[Tuple[str, str]] = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        code = str(row.get(code_col, "")).strip()
        if not name:
            continue
        concepts.append((name, code))
    return concepts


def download_concept_components(
    output_dir: Optional[str] = None,
    resume: bool = True,
    max_items: Optional[int] = None,
    sleep_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds
    if max_retries is not None:
        cfg.max_retries = max_retries
    if backoff_factor is not None:
        cfg.backoff_factor = backoff_factor

    ak_root = _akshare_root(output_dir)
    concept_dir = ak_root / "concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)
    component_dir = concept_dir / "components"
    component_dir.mkdir(parents=True, exist_ok=True)

    concept_list_path = concept_dir / "concept_list.parquet"
    if not concept_list_path.exists():
        concept_list_path = download_concept_list(
            output_dir=output_dir,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

    concept_df = pd.read_parquet(concept_list_path)
    concepts = _extract_concept_fields(concept_df)
    if not concepts:
        raise RuntimeError("概念列表为空，无法下载成分股")

    state_path = cfg.state_dir / "akshare_concept_components.json"
    state = _load_state(state_path) if resume else {}
    start_index = int(state.get("next_index", 0)) if resume else 0
    if start_index >= len(concepts):
        start_index = 0

    processed = 0
    for idx in range(start_index, len(concepts)):
        name, code = concepts[idx]
        safe_name = _safe_filename(f"{code}_{name}" if code else name)
        target_path = component_dir / f"{safe_name}.parquet"

        logger.info(f"下载概念成分: {name} ({code}) [{idx + 1}/{len(concepts)}]")
        df = request_with_retry(lambda: ak.stock_board_concept_cons_em(symbol=name), cfg)
        if df is None or df.empty:
            logger.warning(f"概念 {name} 无成分股数据")
        else:
            df.to_parquet(target_path, index=False)
            logger.info(f"保存成分股: {target_path}")

        _save_state(
            state_path,
            {
                "next_index": idx + 1,
                "concept_count": len(concepts),
            },
        )

        processed += 1
        if max_items and processed >= max_items:
            logger.info(f"达到 max_items={max_items}，停止")
            break

        time.sleep(cfg.sleep_seconds)

    return component_dir


def _normalize_symbol(value: str) -> str:
    value = value.strip().upper()
    if "." in value:
        value = value.split(".", 1)[0]
    digits = "".join(ch for ch in value if ch.isdigit())
    return digits if digits else value


def _load_stock_list(stock_list_csv: Optional[str]) -> List[str]:
    if not stock_list_csv:
        return ["000001", "000002"]

    path = Path(stock_list_csv)
    if not path.exists():
        raise FileNotFoundError(f"stock_list_csv 不存在: {stock_list_csv}")

    df = pd.read_csv(path)
    for column in ("ts_code", "symbol", "code"):
        if column in df.columns:
            values = df[column].dropna().astype(str).tolist()
            return [_normalize_symbol(v) for v in values if str(v).strip()]

    first_col = df.columns[0]
    values = df[first_col].dropna().astype(str).tolist()
    return [_normalize_symbol(v) for v in values if str(v).strip()]


def download_stock_hist(
    start_date: str,
    end_date: str,
    output_dir: Optional[str] = None,
    stock_list_csv: Optional[str] = None,
    resume: bool = True,
    max_items: Optional[int] = None,
    sleep_seconds: Optional[float] = None,
    adjust: str = "qfq",
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
) -> Path:
    cfg = load_config()
    if sleep_seconds is not None:
        cfg.sleep_seconds = sleep_seconds
    if max_retries is not None:
        cfg.max_retries = max_retries
    if backoff_factor is not None:
        cfg.backoff_factor = backoff_factor

    ak_root = _akshare_root(output_dir)
    hist_dir = ak_root / "stock_hist"
    hist_dir.mkdir(parents=True, exist_ok=True)

    start = _to_yyyymmdd(start_date)
    end = _to_yyyymmdd(end_date)
    symbols = _load_stock_list(stock_list_csv)
    if not symbols:
        raise RuntimeError("股票列表为空")

    state_path = cfg.state_dir / "akshare_stock_hist.json"
    state = _load_state(state_path) if resume else {}
    start_index = int(state.get("next_index", 0)) if resume else 0
    if start_index >= len(symbols):
        start_index = 0

    processed = 0
    for idx in range(start_index, len(symbols)):
        symbol = symbols[idx]
        logger.info(f"下载个股历史: {symbol} [{idx + 1}/{len(symbols)}]")
        df = request_with_retry(
            lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
                adjust=adjust,
            ),
            cfg,
        )
        if df is None or df.empty:
            logger.warning(f"{symbol} 无历史数据")
        else:
            if "日期" in df.columns:
                df["日期"] = df["日期"].astype(str).str.replace("-", "")
            target_path = hist_dir / f"{symbol}_{start}_{end}.parquet"
            df.to_parquet(target_path, index=False)
            logger.info(f"保存历史数据: {target_path}")

        _save_state(
            state_path,
            {
                "next_index": idx + 1,
                "symbol_count": len(symbols),
                "start_date": start,
                "end_date": end,
                "adjust": adjust,
            },
        )

        processed += 1
        if max_items and processed >= max_items:
            logger.info(f"达到 max_items={max_items}，停止")
            break

        time.sleep(cfg.sleep_seconds)

    return hist_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Akshare 数据下载工具（统一入口）")
    parser.add_argument(
        "--action",
        required=True,
        choices=["concept_list", "concept_components", "stock_hist"],
        help="执行动作",
    )
    parser.add_argument("--output-dir", help="输出根目录（默认使用 config/base.yaml）")
    parser.add_argument("--sleep-seconds", type=float, help="请求间隔秒数（覆盖配置）")
    parser.add_argument("--max-retries", type=int, help="最大重试次数（覆盖配置）")
    parser.add_argument("--backoff-factor", type=float, help="退避系数（覆盖配置）")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--max-items", type=int, help="最多处理条数（测试用）")
    parser.add_argument("--start-date", help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", help="结束日期 YYYYMMDD")
    parser.add_argument("--stock-list-csv", help="股票列表CSV（含 ts_code/symbol/code 任一列）")
    parser.add_argument("--adjust", default="qfq", help="复权方式（默认 qfq）")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.action == "concept_list":
        download_concept_list(
            output_dir=args.output_dir,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
            backoff_factor=args.backoff_factor,
        )
    elif args.action == "concept_components":
        download_concept_components(
            output_dir=args.output_dir,
            resume=args.resume,
            max_items=args.max_items,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
            backoff_factor=args.backoff_factor,
        )
    elif args.action == "stock_hist":
        if not args.start_date or not args.end_date:
            raise SystemExit("stock_hist 需要 --start-date 和 --end-date")
        download_stock_hist(
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            stock_list_csv=args.stock_list_csv,
            resume=args.resume,
            max_items=args.max_items,
            sleep_seconds=args.sleep_seconds,
            adjust=args.adjust,
            max_retries=args.max_retries,
            backoff_factor=args.backoff_factor,
        )


if __name__ == "__main__":
    # 示例：概念列表
    # download_concept_list(output_dir="/tmp/sage_data")

    # 示例：概念成分（可设置 max_items 便于测试）
    # download_concept_components(output_dir="/tmp/sage_data", resume=True, max_items=20)

    # 示例：个股历史（日线，YYYYMMDD）
    # download_stock_hist(
    #     start_date="20240101",
    #     end_date="20240131",
    #     stock_list_csv="data/raw/tushare/filtered_stocks_list.csv",
    #     output_dir="/tmp/sage_data",
    #     resume=True,
    # )

    main()
