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
import hashlib
import json
import re
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
    log_task_summary,
    setup_logger,
)

add_project_root()

DATE_LIKE_COLUMNS = {
    "trade_date",
    "ann_date",
    "end_date",
    "cal_date",
    "f_ann_date",
    "month",
    "period",
    "list_date",
    "delist_date",
}


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


def _load_base_download_config(project_root: Path) -> Dict[str, Any]:
    config_path = project_root / "config" / "base.yaml"
    if not config_path.exists() or yaml is None:
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        return (payload.get("data") or {}).get("download") or {}
    except Exception:
        return {}


def _parse_date(value: str, fmt: str = "%Y%m%d") -> datetime:
    return datetime.strptime(value, fmt)


def _format_date(value: datetime, fmt: str = "%Y%m%d") -> str:
    return value.strftime(fmt)


def _month_windows(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    current = datetime(start.year, start.month, 1)
    while current <= end:
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = next_month - timedelta(days=1)
        window_start = current if current > start else start
        window_end = month_end if month_end < end else end
        yield window_start, window_end
        current = next_month


def _daily_windows(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    current = start
    while current <= end:
        yield current, current
        current += timedelta(days=1)


def _quarter_windows(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    quarter_starts = [(1, 1), (4, 1), (7, 1), (10, 1)]
    quarter_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
    current_year = start.year
    current_q = (start.month - 1) // 3
    while True:
        m_start, d_start = quarter_starts[current_q]
        m_end, d_end = quarter_ends[current_q]
        q_start = datetime(current_year, m_start, d_start)
        q_end = datetime(current_year, m_end, d_end)
        if q_end < start:
            current_q += 1
            if current_q > 3:
                current_q = 0
                current_year += 1
            continue
        if q_start > end:
            break
        window_start = q_start if q_start > start else start
        window_end = q_end if q_end < end else end
        yield window_start, window_end
        current_q += 1
        if current_q > 3:
            current_q = 0
            current_year += 1


def _year_windows(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    current = datetime(start.year, 1, 1)
    while current.year <= end.year:
        year_end = datetime(current.year, 12, 31)
        if year_end > end:
            year_end = end
        yield current, year_end
        current = datetime(current.year + 1, 1, 1)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    logger.info(message)


logger = setup_logger("tushare_downloader", module="data")


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


def _write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    _ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


def _safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        _write_parquet_atomic(df, path)
        return
    except Exception as exc:
        message = str(exc)
        retry_df = df.copy()
        fallback_cols = set()

        # pyarrow 常见报错："Conversion failed for column trade_date with type object"
        m = re.search(r"column\s+([A-Za-z0-9_]+)", message)
        if m:
            fallback_cols.add(m.group(1))

        # 日期类字段最容易出现 int/str 混合，统一转为文本。
        fallback_cols.update(col for col in DATE_LIKE_COLUMNS if col in retry_df.columns)
        for col in fallback_cols:
            if col in retry_df.columns:
                retry_df[col] = _to_text_series(retry_df[col])
        if fallback_cols:
            _write_parquet_atomic(retry_df, path)
            return
        raise


def _to_text_series(series: pd.Series) -> pd.Series:
    def _normalize(value: Any) -> Optional[str]:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if text.endswith(".0"):
            text = text[:-2]
        return text

    return series.map(_normalize)


@dataclass
class TaskConfig:
    name: str
    api: str
    mode: str
    output: str
    output_aliases: Optional[List[str]] = None
    state: Optional[str] = None
    start_field: Optional[str] = None
    end_field: Optional[str] = None
    split: Optional[str] = None
    pagination: Optional[str] = None
    limit: int = 3000
    dedup_keys: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None
    date_format: str = "%Y%m%d"
    sleep_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    backoff_seconds: Optional[int] = None
    continue_on_error: bool = True
    list_file: Optional[str] = None
    list_format: Optional[str] = None
    list_column: Optional[str] = None
    list_columns: Optional[List[str]] = None
    list_items: Optional[List[Any]] = None
    list_param: Optional[str] = None
    filter_field: Optional[str] = None
    filter_list_file: Optional[str] = None
    filter_list_format: Optional[str] = None
    filter_list_column: Optional[str] = None
    filter_list_columns: Optional[List[str]] = None
    filter_list_items: Optional[List[Any]] = None
    skip_coverage_check: bool = False
    quarters: Optional[List[str]] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None

    @classmethod
    def from_dict(cls, name: str, payload: Dict[str, Any]) -> "TaskConfig":
        return cls(
            name=name,
            api=payload["api"],
            mode=payload.get("mode", "single"),
            output=payload["output"],
            output_aliases=payload.get("output_aliases") or None,
            state=payload.get("state"),
            start_field=payload.get("start_field"),
            end_field=payload.get("end_field"),
            split=payload.get("split"),
            pagination=payload.get("pagination"),
            limit=int(payload.get("limit", 3000)),
            dedup_keys=list(payload.get("dedup_keys", [])) or None,
            params=payload.get("params") or {},
            date_format=payload.get("date_format", "%Y%m%d"),
            sleep_seconds=payload.get("sleep_seconds"),
            max_retries=payload.get("max_retries"),
            backoff_seconds=payload.get("backoff_seconds"),
            continue_on_error=payload.get("continue_on_error", True),
            list_file=payload.get("list_file"),
            list_format=payload.get("list_format"),
            list_column=payload.get("list_column"),
            list_columns=payload.get("list_columns"),
            list_items=payload.get("list_items"),
            list_param=payload.get("list_param"),
            filter_field=payload.get("filter_field"),
            filter_list_file=payload.get("filter_list_file"),
            filter_list_format=payload.get("filter_list_format"),
            filter_list_column=payload.get("filter_list_column"),
            filter_list_columns=payload.get("filter_list_columns"),
            filter_list_items=payload.get("filter_list_items"),
            skip_coverage_check=bool(payload.get("skip_coverage_check", False)),
            quarters=payload.get("quarters"),
            start_year=payload.get("start_year"),
            end_year=payload.get("end_year"),
        )


class TushareClient:
    def __init__(
        self,
        token: Optional[str] = None,
        sleep_seconds: int = 40,
        disable_proxy_flag: bool = True,
        max_retries: int = 3,
        backoff_seconds: int = 30,
    ) -> None:
        if disable_proxy_flag:
            disable_proxy()
        import tushare as ts  # local import

        self.pro = ts.pro_api(get_tushare_token(token))
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def request(self, api: str, params: Dict[str, Any]) -> pd.DataFrame:
        for attempt in range(self.max_retries):
            try:
                func = getattr(self.pro, api)
                df = func(**params)
                time.sleep(self.sleep_seconds)
                return df
            except Exception as exc:
                if attempt >= self.max_retries - 1:
                    raise
                wait = (attempt + 1) * self.backoff_seconds
                _log(f"请求失败: {exc}，{wait}s 后重试...")
                time.sleep(wait)
        return pd.DataFrame()

    def request_paged(self, api: str, params: Dict[str, Any], limit: int = 3000) -> pd.DataFrame:
        offset = 0
        frames: List[pd.DataFrame] = []
        while True:
            page_params = dict(params)
            page_params["offset"] = offset
            page_params["limit"] = limit
            _log(f"分页请求 {api} offset={offset} limit={limit}")
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


def _resolve_output_aliases(task: TaskConfig, output_root: Path) -> List[Path]:
    aliases: List[Path] = []
    for alias in task.output_aliases or []:
        if not alias:
            continue
        alias_path = _resolve_output(alias, output_root)
        if alias_path not in aliases:
            aliases.append(alias_path)
    return aliases


def _resolve_state(path_str: Optional[str], project_root: Optional[Path] = None) -> Optional[Path]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_absolute() and project_root is not None:
        path = project_root / path
    if path.is_dir():
        return path / "state.json"
    return path


def _default_state_path(task_name: str) -> Path:
    return get_data_path("states", "tushare", f"{task_name}.json", ensure=True)


def _iter_windows(
    mode: str,
    split: Optional[str],
    start: datetime,
    end: datetime,
    date_format: str,
) -> Iterable[tuple[str, str]]:
    if mode == "single":
        yield _format_date(start, date_format), _format_date(end, date_format)
        return
    if split == "day":
        for s, e in _daily_windows(start, end):
            yield _format_date(s, date_format), _format_date(e, date_format)
        return
    if split == "month":
        for s, e in _month_windows(start, end):
            yield _format_date(s, date_format), _format_date(e, date_format)
        return
    if split == "quarter":
        for s, e in _quarter_windows(start, end):
            yield _format_date(s, date_format), _format_date(e, date_format)
        return
    if split == "year":
        for s, e in _year_windows(start, end):
            yield _format_date(s, date_format), _format_date(e, date_format)
        return
    yield _format_date(start, date_format), _format_date(end, date_format)


def _load_list_items(task: TaskConfig, project_root: Path) -> List[Any]:
    if task.list_items:
        return task.list_items
    if not task.list_file:
        return []
    list_path = Path(task.list_file)
    if not list_path.is_absolute():
        list_path = project_root / list_path
    if not list_path.exists():
        raise FileNotFoundError(f"列表文件不存在: {list_path}")

    fmt = (task.list_format or list_path.suffix.replace(".", "")).lower()
    if fmt in ("csv", "txt"):
        df = pd.read_csv(list_path)
    elif fmt in ("parquet", "pq"):
        df = pd.read_parquet(list_path)
    else:
        df = pd.read_csv(list_path)

    columns = task.list_columns or ([task.list_column] if task.list_column else [])
    for col in columns:
        if col and col in df.columns:
            return df[col].dropna().astype(str).unique().tolist()

    raise ValueError(f"列表文件缺少列: {columns}")


def _load_filter_items(task: TaskConfig, project_root: Path) -> Optional[set[str]]:
    if task.filter_list_items:
        return {str(v) for v in task.filter_list_items if v is not None}
    if not task.filter_list_file:
        return None
    list_path = Path(task.filter_list_file)
    if not list_path.is_absolute():
        list_path = project_root / list_path
    if not list_path.exists():
        raise FileNotFoundError(f"过滤列表文件不存在: {list_path}")

    fmt = (task.filter_list_format or list_path.suffix.replace(".", "")).lower()
    if fmt in ("csv", "txt"):
        df = pd.read_csv(list_path)
    elif fmt in ("parquet", "pq"):
        df = pd.read_parquet(list_path)
    else:
        df = pd.read_csv(list_path)

    columns = task.filter_list_columns or ([task.filter_list_column] if task.filter_list_column else [])
    for col in columns:
        if col and col in df.columns:
            return set(df[col].dropna().astype(str).unique().tolist())
    raise ValueError(f"过滤列表文件缺少列: {columns}")


def _iter_quarters(task: TaskConfig) -> List[str]:
    start_year = task.start_year or datetime.now().year
    end_year = task.end_year or start_year
    quarters = task.quarters or ["0331", "0630", "0930", "1231"]
    periods: List[str] = []
    for year in range(start_year, end_year + 1):
        for q in quarters:
            periods.append(f"{year}{q}")
    return periods


def _item_key(item: Any) -> str:
    if item is None:
        return "None"
    if isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False, sort_keys=True)
    return str(item)


def _build_signature(task_name: str, windows: List[tuple[str, str]], items: List[Any]) -> str:
    payload = {
        "task": task_name,
        "windows": windows,
        "items": [_item_key(i) for i in items],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def run_task(
    task: TaskConfig,
    start_date: Optional[str],
    end_date: Optional[str],
    output_root: Path,
    sleep_seconds: int,
    resume: bool,
    dry_run: bool,
    retry_failed: bool = False,
    disable_proxy_flag: bool = True,
) -> None:
    task_start_time = time.time()
    failure_reason = None
    _log(f"开始任务: {task.name}, start_date={start_date}, end_date={end_date}, resume={resume}")
    output_path = _resolve_output(task.output, output_root)
    output_aliases = _resolve_output_aliases(task, output_root)
    project_root = add_project_root()
    state_path = _resolve_state(task.state, project_root=project_root) if task.state else _default_state_path(task.name)

    start_dt = datetime.now()
    end_dt = start_dt

    if task.mode == "single":
        start_date = start_date or datetime.now().strftime(task.date_format)
        end_date = end_date or start_date
        start_dt = _parse_date(start_date, task.date_format)
        end_dt = _parse_date(end_date, task.date_format)
    elif task.mode == "date_range":
        if not start_date or not end_date:
            raise ValueError("该任务需要 --start-date 与 --end-date（YYYYMMDD）")
        start_dt = _parse_date(start_date, task.date_format)
        end_dt = _parse_date(end_date, task.date_format)
    elif task.mode == "list":
        if start_date and end_date:
            start_dt = _parse_date(start_date, task.date_format)
            end_dt = _parse_date(end_date, task.date_format)

    last_end = None
    cursor_index = None
    state_signature = None
    failed_items: List[Dict[str, Any]] = []
    if resume and state_path:
        state = _load_state(state_path)
        last_end = state.get("last_end")
        cursor_index = state.get("cursor_index")
        state_signature = state.get("signature")
        failed_items = state.get("failed", [])
        _log(
            f"读取断点: state={state_path}, last_end={last_end}, cursor_index={cursor_index}, failed={len(failed_items)}"
        )

    backfill_only = False
    if last_end and end_date:
        try:
            last_end_dt = _parse_date(last_end, task.date_format)
            if end_dt <= last_end_dt:
                backfill_only = True
                _log(f"检测到回补模式: end_date={end_date} <= last_end={last_end}")
        except Exception:
            _log(f"⚠️  断点 last_end={last_end} 解析失败，跳过回补判断")

    if task.mode in ("year_quarters",) or (task.mode == "list" and not start_date and not end_date):
        windows = [("", "")]
    else:
        windows = list(_iter_windows(task.mode, task.split, start_dt, end_dt, task.date_format))
    if last_end and not backfill_only:
        windows = [win for win in windows if win[1] > last_end]
    _log(f"待处理窗口数量: {len(windows)}")

    if dry_run:
        _log(f"任务 {task.name} 将处理 {len(windows)} 个窗口")
        log_task_summary(
            logger,
            task_name=task.name,
            window=f"{start_date or ''}~{end_date or ''}",
            elapsed_s=time.time() - task_start_time,
            error=None,
        )
        return

    _log("初始化 Tushare 客户端...")
    client = TushareClient(
        sleep_seconds=sleep_seconds,
        disable_proxy_flag=disable_proxy_flag,
        max_retries=task.max_retries or 3,
        backoff_seconds=task.backoff_seconds or 30,
    )
    _log("Tushare 客户端初始化完成")

    existing_frames: List[pd.DataFrame] = []
    if output_path.exists():
        _log(f"读取已有输出文件: {output_path}")
        existing_frames.append(pd.read_parquet(output_path))
    for alias_path in output_aliases:
        if alias_path.exists():
            _log(f"读取已有别名输出: {alias_path}")
            existing_frames.append(pd.read_parquet(alias_path))
    existing_sources = len(existing_frames)
    if existing_frames:
        existing = pd.concat(existing_frames, ignore_index=True)
        if task.dedup_keys:
            valid_keys = [k for k in task.dedup_keys if k in existing.columns]
            if valid_keys:
                for key in valid_keys:
                    if key in DATE_LIKE_COLUMNS:
                        existing[key] = _to_text_series(existing[key])
                existing = existing.drop_duplicates(subset=valid_keys)
        _log(f"已有数据行数: {len(existing)}")
    else:
        existing = pd.DataFrame()
        _log(f"输出文件不存在，将新建: {output_path}")

    list_items = _load_list_items(task, project_root) if task.mode == "list" else [None]
    if task.mode == "list" and not list_items:
        raise ValueError(f"{task.name} 缺少 list_items 或 list_file")

    if task.mode == "year_quarters":
        list_items = _iter_quarters(task)

    filter_items = None
    if task.filter_field:
        filter_items = _load_filter_items(task, project_root)
        if filter_items is None:
            _log(f"⚠️  未加载过滤列表，跳过过滤: {task.name}")

    work_items: List[Dict[str, Any]] = []
    for item in list_items:
        for win_start, win_end in windows:
            work_items.append(
                {
                    "item": item,
                    "item_key": _item_key(item),
                    "win_start": win_start,
                    "win_end": win_end,
                }
            )
    current_signature = _build_signature(task.name, windows, list_items)

    if resume and state_signature and cursor_index is not None:
        if state_signature == current_signature:
            work_items = work_items[cursor_index + 1 :]
        else:
            _log("断点签名不一致，回退到 last_end 过滤")
            if last_end and not backfill_only:
                work_items = [item for item in work_items if item["win_end"] > last_end]

    if resume and failed_items:
        if retry_failed:
            _log(f"优先重试失败窗口: {len(failed_items)}")
        else:
            _log(f"检测到失败窗口，自动重试: {len(failed_items)}")
        retry_items = []
        for failure in failed_items:
            retry_items.append(
                {
                    "item": failure.get("item"),
                    "item_key": failure.get("item_key", _item_key(failure.get("item"))),
                    "win_start": failure.get("win_start"),
                    "win_end": failure.get("win_end"),
                }
            )
        work_items = retry_items + work_items
        failed_items = []

    def _state_last_end(win_end: str) -> str:
        nonlocal last_end
        if not backfill_only:
            last_end = win_end
            return win_end
        if last_end is None or win_end > last_end:
            last_end = win_end
        return last_end

    total_steps = len(work_items)
    if total_steps == 0 and existing_sources > 0:
        _log("无待处理窗口，写回合并后的历史数据")
        _safe_to_parquet(existing, output_path)
        for alias_path in output_aliases:
            if alias_path == output_path:
                continue
            _safe_to_parquet(existing, alias_path)
    signature = current_signature
    try:
        for step, work in enumerate(work_items, 1):
            item = work["item"]
            win_start = work["win_start"]
            win_end = work["win_end"]
            params = dict(task.params or {})
            if task.start_field:
                params[task.start_field] = win_start
            if task.end_field:
                params[task.end_field] = win_end
            if item is not None:
                if isinstance(item, dict):
                    params.update(item)
                elif task.list_param:
                    params[task.list_param] = item

            window_desc = f"{win_start} ~ {win_end}".strip()
            if item is not None:
                if window_desc.strip(" ~") == "":
                    window_desc = str(item)
                else:
                    window_desc = f"{window_desc} | {task.list_param or 'item'}={item}"
            _log(f"[{step}/{total_steps}] {task.name} {window_desc} ...")
            try:
                if task.pagination == "offset":
                    df = client.request_paged(task.api, params, limit=task.limit)
                else:
                    df = client.request(task.api, params)
            except Exception as exc:
                _log(f"❌ 请求失败: {exc}")
                if state_path:
                    failed_items.append(
                        {
                            "item": item,
                            "item_key": work["item_key"],
                            "win_start": win_start,
                            "win_end": win_end,
                            "error": str(exc),
                        }
                    )
                    _save_state(
                        state_path,
                        {
                            "last_end": _state_last_end(win_end),
                            "cursor_index": step - 1,
                            "signature": signature,
                            "failed": failed_items[-200:],
                        },
                    )
                if task.continue_on_error:
                    continue
                raise

            if df is None or df.empty:
                _log("无数据")
                if state_path:
                    _save_state(
                        state_path,
                        {
                            "last_end": _state_last_end(win_end),
                            "cursor_index": step - 1,
                            "signature": signature,
                            "failed": failed_items[-200:],
                        },
                    )
                continue

            if filter_items and task.filter_field in df.columns:
                before = len(df)
                df = df[df[task.filter_field].astype(str).isin(filter_items)]
                _log(f"过滤 {task.filter_field}: {before} -> {len(df)}")
            elif filter_items and task.filter_field not in df.columns:
                _log(f"⚠️  过滤字段不存在: {task.filter_field}")

            if existing.empty:
                combined = df
            else:
                combined = pd.concat([existing, df], ignore_index=True)
                if task.dedup_keys:
                    # 只对存在的列进行去重
                    valid_dedup_keys = [k for k in task.dedup_keys if k in combined.columns]
                    if valid_dedup_keys:
                        for key in valid_dedup_keys:
                            if key in DATE_LIKE_COLUMNS:
                                combined[key] = _to_text_series(combined[key])
                        combined = combined.drop_duplicates(subset=valid_dedup_keys)
                    else:
                        _log(f"⚠️  去重键 {task.dedup_keys} 在数据中不存在，跳过去重")

            _safe_to_parquet(combined, output_path)
            for alias_path in output_aliases:
                if alias_path == output_path:
                    continue
                _safe_to_parquet(combined, alias_path)
            existing = combined

            if state_path:
                _save_state(
                    state_path,
                    {
                        "last_end": _state_last_end(win_end),
                        "cursor_index": step - 1,
                        "signature": signature,
                        "failed": failed_items[-200:],
                    },
                )

        if output_path.exists():
            _log(f"✅ 输出: {output_path}")
        for alias_path in output_aliases:
            if alias_path.exists() and alias_path != output_path:
                _log(f"✅ 输出: {alias_path}")
    except Exception as exc:
        failure_reason = str(exc)
        raise
    finally:
        log_task_summary(
            logger,
            task_name=task.name,
            window=f"{start_date or ''}~{end_date or ''}",
            elapsed_s=time.time() - task_start_time,
            error=failure_reason,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tushare 下载器（理想版）")
    parser.add_argument("--task", required=True, help="任务名，见 config/tushare_tasks.yaml")
    parser.add_argument("--config", default="config/tushare_tasks.yaml")
    parser.add_argument("--start-date", help="YYYYMMDD")
    parser.add_argument("--end-date", help="YYYYMMDD")
    parser.add_argument("--sleep-seconds", type=int)
    parser.add_argument("--output-root", help="覆盖输出根目录")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-proxy", dest="disable_proxy", action="store_true", help="禁用代理（默认）")
    parser.add_argument("--enable-proxy", dest="disable_proxy", action="store_false", help="启用代理")
    parser.set_defaults(disable_proxy=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = _load_yaml(config_path)
    tasks = cfg.get("tasks") or {}
    if args.task not in tasks:
        raise KeyError(f"任务未定义: {args.task}")
    task = TaskConfig.from_dict(args.task, tasks[args.task])

    output_root = Path(args.output_root) if args.output_root else get_tushare_root(ensure=True)
    base_download = _load_base_download_config(add_project_root())
    sleep_seconds = args.sleep_seconds or task.sleep_seconds or int(base_download.get("sleep_seconds", 40))
    if task.max_retries is None:
        task.max_retries = int(base_download.get("max_retries", 3))
    if task.backoff_seconds is None:
        backoff_factor = float(base_download.get("backoff_factor", 2.0))
        task.backoff_seconds = max(10, int(sleep_seconds * backoff_factor))

    run_task(
        task=task,
        start_date=args.start_date,
        end_date=args.end_date,
        output_root=output_root,
        sleep_seconds=sleep_seconds,
        resume=args.resume,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        disable_proxy_flag=args.disable_proxy,
    )


if __name__ == "__main__":
    main()
