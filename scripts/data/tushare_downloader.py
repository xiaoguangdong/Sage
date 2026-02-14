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


def _parse_date(value: str, fmt: str = "%Y%m%d") -> datetime:
    return datetime.strptime(value, fmt)


def _format_date(value: datetime, fmt: str = "%Y%m%d") -> str:
    return value.strftime(fmt)


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


def _retry_call(func, retries: int = 3, backoff: int = 30, sleep_seconds: int = 1):
    for attempt in range(retries):
        try:
            result = func()
            time.sleep(sleep_seconds)
            return result
        except Exception as exc:
            if attempt >= retries - 1:
                raise
            wait = (attempt + 1) * backoff
            print(f"  请求失败: {exc}，{wait}s 后重试...")
            time.sleep(wait)


@dataclass
class TaskConfig:
    name: str
    api: str
    mode: str
    output: str
    provider: str = "tushare"
    state: Optional[str] = None
    start_field: Optional[str] = None
    end_field: Optional[str] = None
    split: Optional[str] = None
    pagination: Optional[str] = None
    limit: int = 3000
    dedup_keys: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None
    date_format: str = "%Y%m%d"
    list_file: Optional[str] = None
    list_format: Optional[str] = None
    list_column: Optional[str] = None
    list_columns: Optional[List[str]] = None
    list_items: Optional[List[Any]] = None
    list_param: Optional[str] = None
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
            provider=payload.get("provider", "tushare"),
            state=payload.get("state"),
            start_field=payload.get("start_field"),
            end_field=payload.get("end_field"),
            split=payload.get("split"),
            pagination=payload.get("pagination"),
            limit=int(payload.get("limit", 3000)),
            dedup_keys=list(payload.get("dedup_keys", [])) or None,
            params=payload.get("params") or {},
            date_format=payload.get("date_format", "%Y%m%d"),
            list_file=payload.get("list_file"),
            list_format=payload.get("list_format"),
            list_column=payload.get("list_column"),
            list_columns=payload.get("list_columns"),
            list_items=payload.get("list_items"),
            list_param=payload.get("list_param"),
            quarters=payload.get("quarters"),
            start_year=payload.get("start_year"),
            end_year=payload.get("end_year"),
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
            page_params["limit"] = limit
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
    if split == "year":
        for s, e in _year_windows(start, end):
            yield _format_date(s, date_format), _format_date(e, date_format)
        return
    yield _format_date(start), _format_date(end)


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


def _iter_quarters(task: TaskConfig) -> List[str]:
    start_year = task.start_year or datetime.now().year
    end_year = task.end_year or start_year
    quarters = task.quarters or ["0331", "0630", "0930", "1231"]
    periods: List[str] = []
    for year in range(start_year, end_year + 1):
        for q in quarters:
            periods.append(f"{year}{q}")
    return periods


def _ak_safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in value.strip())


def _ak_load_concepts(path: Path) -> List[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"概念列表不存在: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    name_col = "板块名称" if "板块名称" in df.columns else "名称"
    code_col = "板块代码" if "板块代码" in df.columns else "代码"
    concepts: List[tuple[str, str]] = []
    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        code = str(row.get(code_col, "")).strip()
        if name:
            concepts.append((name, code))
    return concepts


def _ak_load_stock_list(list_path: Optional[str], project_root: Path) -> List[str]:
    if not list_path:
        return ["000001", "000002"]
    path = Path(list_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"stock_list_csv 不存在: {path}")
    df = pd.read_csv(path)
    for column in ("ts_code", "symbol", "code"):
        if column in df.columns:
            values = df[column].dropna().astype(str).tolist()
            return [v.split(".", 1)[0] for v in values if str(v).strip()]
    first_col = df.columns[0]
    values = df[first_col].dropna().astype(str).tolist()
    return [v.split(".", 1)[0] for v in values if str(v).strip()]


def _run_akshare_task(
    task: TaskConfig,
    start_date: Optional[str],
    end_date: Optional[str],
    output_root: Path,
    sleep_seconds: int,
    resume: bool,
    dry_run: bool,
    project_root: Path,
) -> None:
    try:
        import akshare as ak  # type: ignore
    except Exception as exc:
        raise RuntimeError("缺少 akshare 依赖，请先安装") from exc

    output_path = _resolve_output(task.output, output_root)
    if dry_run:
        print(f"任务 {task.name} (akshare) 输出到 {output_path}")
        return

    if task.name == "ak_concept_list":
        df = _retry_call(lambda: ak.stock_board_concept_name_em(), sleep_seconds=sleep_seconds)
        _safe_to_parquet(df, output_path)
        print(f"✅ 输出: {output_path}")
        return

    if task.name == "ak_concept_components":
        state_path = _resolve_state(task.state) if task.state else None
        state = _load_state(state_path) if (resume and state_path) else {}
        start_index = int(state.get("next_index", 0))

        list_path = _resolve_output(task.params.get("concept_list_path", "akshare/concepts/concept_list.parquet"), output_root)
        concepts = _ak_load_concepts(list_path)
        component_dir = output_path.parent
        component_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(start_index, len(concepts)):
            name, code = concepts[idx]
            print(f"[{idx+1}/{len(concepts)}] 概念成分: {name}")
            df = _retry_call(lambda: ak.stock_board_concept_cons_em(symbol=name), sleep_seconds=sleep_seconds)
            if df is not None and not df.empty:
                df["concept_name"] = name
                df["concept_code"] = code
                filename = component_dir / f"{_ak_safe_filename(name)}.parquet"
                _safe_to_parquet(df, filename)
            if state_path:
                _save_state(state_path, {"next_index": idx + 1, "total": len(concepts)})
        return

    if task.name == "ak_stock_hist":
        if not start_date or not end_date:
            raise ValueError("该任务需要 --start-date 与 --end-date（YYYYMMDD）")
        adjust = (task.params or {}).get("adjust", "qfq")
        symbols = _ak_load_stock_list(task.list_file, project_root)
        state_path = _resolve_state(task.state) if task.state else None
        state = _load_state(state_path) if (resume and state_path) else {}
        start_index = int(state.get("next_index", 0))

        start_fmt = start_date
        end_fmt = end_date
        for idx in range(start_index, len(symbols)):
            symbol = symbols[idx]
            print(f"[{idx+1}/{len(symbols)}] 个股历史: {symbol}")
            df = _retry_call(
                lambda: ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_fmt,
                    end_date=end_fmt,
                    adjust=adjust,
                ),
                sleep_seconds=sleep_seconds,
            )
            if df is not None and not df.empty:
                if "日期" in df.columns:
                    df["日期"] = df["日期"].astype(str).str.replace("-", "")
                target = output_path.parent / f"{symbol}_{start_fmt}_{end_fmt}.parquet"
                _safe_to_parquet(df, target)
            if state_path:
                _save_state(state_path, {"next_index": idx + 1, "symbol_count": len(symbols)})
        return

    raise ValueError(f"未知 akshare 任务: {task.name}")


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

    project_root = add_project_root()

    if task.provider == "akshare":
        _run_akshare_task(task, start_date, end_date, output_root, sleep_seconds, resume, dry_run, project_root)
        return

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
    if resume and state_path:
        state = _load_state(state_path)
        last_end = state.get("last_end")

    if task.mode in ("year_quarters",) or (task.mode == "list" and not start_date and not end_date):
        windows = [("", "")]
    else:
        windows = list(_iter_windows(task.mode, task.split, start_dt, end_dt, task.date_format))
    if last_end:
        windows = [win for win in windows if win[1] > last_end]

    if dry_run:
        print(f"任务 {task.name} 将处理 {len(windows)} 个窗口")
        return

    client = TushareClient(sleep_seconds=sleep_seconds)

    existing = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()

    list_items = _load_list_items(task, project_root) if task.mode == "list" else [None]
    if task.mode == "list" and not list_items:
        raise ValueError(f"{task.name} 缺少 list_items 或 list_file")

    if task.mode == "year_quarters":
        list_items = _iter_quarters(task)

    total_steps = len(windows) * len(list_items)
    step = 0
    for item in list_items:
        for win_start, win_end in windows:
            step += 1
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

            print(f"[{step}/{total_steps}] {task.name} {win_start} ~ {win_end} ...")
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

    output_root = Path(args.output_root) if args.output_root else get_data_path("raw", ensure=True)

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
def _year_windows(start: datetime, end: datetime) -> Iterable[tuple[datetime, datetime]]:
    current = datetime(start.year, 1, 1)
    while current.year <= end.year:
        year_end = datetime(current.year, 12, 31)
        if year_end > end:
            year_end = end
        yield current, year_end
        current = datetime(current.year + 1, 1, 1)
