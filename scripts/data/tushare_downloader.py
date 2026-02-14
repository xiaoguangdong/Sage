#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理想版 Tushare 下载器（统一入口 + 配置驱动 + 断点续传 + 重试/限速）

用法示例：
  python scripts/data/tushare_downloader.py --task ths_index
  python scripts/data/tushare_downloader.py --task ths_daily --start-date 20200101 --end-date 20251231 --resume
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from scripts.data._shared.runtime import (
    add_project_root,
    disable_proxy,
    get_data_path,
    get_tushare_root,
    get_tushare_token,
)


add_project_root()


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("缺少 PyYAML 依赖，请先安装：pip install pyyaml")
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d")


def _format_date(value: datetime) -> str:
    return value.strftime("%Y%m%d")


def _month_windows(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    current = datetime(start.year, start.month, 1)
    while current <= end:
        month_end = (current.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        if month_end > end:
            month_end = end
        yield current, month_end
        current = month_end + timedelta(days=1)
        current = datetime(current.year, current.month, 1)


def _daily_windows(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    current = start
    while current <= end:
        yield current, current
        current += timedelta(days=1)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if path.is_dir():
        raise IsADirectoryError(f"状态文件路径是目录: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    _ensure_parent(path)
    try:
        df.to_parquet(path, index=False)
        return
    except Exception as exc:
        message = str(exc)
        col_name = None
        if "column" in message:
            parts = message.split("column")
            if len(parts) > 1:
                col_name = parts[-1].strip().strip("'").strip('"')
        if col_name and col_name in df.columns:
            df[col_name] = df[col_name].astype(str)
            df.to_parquet(path, index=False)
            return
        raise


@dataclass
class TaskConfig:
    name: str
    api: str
    mode: str
    output: str
    state: Optional[str] = None
    start_field: Optional[str] = None
    end_field: Optional[str] = None
    split: Optional[str] = None
    pagination: Optional[str] = None
    limit: int = 3000
    dedup_keys: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, name: str, payload: Dict[str, Any]) -> "TaskConfig":
        return cls(
            name=name,
            api=payload["api"],
            mode=payload.get("mode", "single"),
            output=payload["output"],
            state=payload.get("state"),
            start_field=payload.get("start_field"),
            end_field=payload.get("end_field"),
            split=payload.get("split"),
            pagination=payload.get("pagination"),
            limit=int(payload.get("limit", 3000)),
            dedup_keys=list(payload.get("dedup_keys", [])) or None,
            params=payload.get("params") or {},
        )


class TushareClient:
    def __init__(self, token: Optional[str] = None, sleep_seconds: int = 40, disable_proxy_flag: bool = True) -> None:
        if disable_proxy_flag:
            disable_proxy()
        import tushare as ts  # local import

        self.pro = ts.pro_api(get_tushare_token(token))
        self.sleep_seconds = sleep_seconds

    def request(self, api: str, params: Dict[str, Any], retries: int = 3, backoff: int = 30) -> pd.DataFrame:
        for attempt in range(retries):
            try:
                func = getattr(self.pro, api)
                df = func(**params)
                time.sleep(self.sleep_seconds)
                return df
            except Exception as exc:
                if attempt >= retries - 1:
                    raise
                wait = (attempt + 1) * backoff
                print(f"  请求失败: {exc}，{wait}s 后重试...")
                time.sleep(wait)
        return pd.DataFrame()

    def request_paged(self, api: str, params: Dict[str, Any], limit: int = 3000) -> pd.DataFrame:
        offset = 0
        frames: List[pd.DataFrame] = []
        while True:
            page_params = dict(params)
            page_params["offset"] = offset
            df = self.request(api, page_params)
            if df is None or df.empty:
                break
            frames.append(df)
            if len(df) < limit:
                break
            offset += len(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _resolve_output(path_str: str, output_root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        return output_root / path
    return path


def _resolve_state(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_dir():
        return path / "state.json"
    return path


def _iter_windows(mode: str, split: Optional[str], start: datetime, end: datetime) -> Iterable[tuple[str, str]]:
    if mode == "single":
        yield _format_date(start), _format_date(end)
        return
    if split == "day":
        for s, e in _daily_windows(start, end):
            yield _format_date(s), _format_date(e)
        return
    if split == "month":
        for s, e in _month_windows(start, end):
            yield _format_date(s), _format_date(e)
        return
    yield _format_date(start), _format_date(end)


def run_task(
    task: TaskConfig,
    start_date: Optional[str],
    end_date: Optional[str],
    output_root: Path,
    sleep_seconds: int,
    resume: bool,
    dry_run: bool,
) -> None:
    output_path = _resolve_output(task.output, output_root)
    state_path = _resolve_state(task.state) if task.state else None

    if task.mode == "single":
        start_date = start_date or datetime.now().strftime("%Y%m%d")
        end_date = end_date or start_date
    else:
        if not start_date or not end_date:
            raise ValueError("该任务需要 --start-date 与 --end-date（YYYYMMDD）")

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    last_end = None
    if resume and state_path:
        state = _load_state(state_path)
        last_end = state.get("last_end")

    windows = list(_iter_windows(task.mode, task.split, start_dt, end_dt))
    if last_end:
        windows = [win for win in windows if win[1] > last_end]

    if dry_run:
        print(f"任务 {task.name} 将处理 {len(windows)} 个窗口")
        return

    client = TushareClient(sleep_seconds=sleep_seconds)

    existing = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
    for idx, (win_start, win_end) in enumerate(windows, start=1):
        params = dict(task.params or {})
        if task.start_field:
            params[task.start_field] = win_start
        if task.end_field:
            params[task.end_field] = win_end
        print(f"[{idx}/{len(windows)}] {task.name} {win_start} ~ {win_end} ...")
        if task.pagination == "offset":
            df = client.request_paged(task.api, params, limit=task.limit)
        else:
            df = client.request(task.api, params)

        if df is None or df.empty:
            print("  无数据")
            if state_path:
                _save_state(state_path, {"last_end": win_end})
            continue

        if existing.empty:
            combined = df
        else:
            combined = pd.concat([existing, df], ignore_index=True)
            if task.dedup_keys:
                combined = combined.drop_duplicates(subset=task.dedup_keys)

        _safe_to_parquet(combined, output_path)
        existing = combined

        if state_path:
            _save_state(state_path, {"last_end": win_end})

    if output_path.exists():
        print(f"✅ 输出: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tushare 下载器（理想版）")
    parser.add_argument("--task", required=True, help="任务名，见 config/tushare_tasks.yaml")
    parser.add_argument("--config", default="config/tushare_tasks.yaml")
    parser.add_argument("--start-date", help="YYYYMMDD")
    parser.add_argument("--end-date", help="YYYYMMDD")
    parser.add_argument("--sleep-seconds", type=int, default=40)
    parser.add_argument("--output-root", help="覆盖输出根目录")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-proxy", action="store_true", default=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = _load_yaml(config_path)
    tasks = cfg.get("tasks") or {}
    if args.task not in tasks:
        raise KeyError(f"任务未定义: {args.task}")
    task = TaskConfig.from_dict(args.task, tasks[args.task])

    output_root = Path(args.output_root) if args.output_root else get_tushare_root(ensure=True)

    if args.disable_proxy:
        disable_proxy()

    run_task(
        task=task,
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=output_root,
        sleep_seconds=args.sleep_seconds,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
