#!/usr/bin/env python3
"""
Daily T-1 data update runner.
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from scripts.data._shared.runtime import add_project_root, get_tushare_root
from scripts.data.tushare_downloader import TaskConfig, _load_yaml, run_task


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _parse_date(value: Optional[str]) -> datetime:
    if value:
        return datetime.strptime(value, "%Y%m%d")
    return datetime.now() - timedelta(days=1)


def _format_date_for_task(task: TaskConfig, dt: datetime) -> str:
    return dt.strftime(task.date_format)


def _load_task_list(cfg_path: Path) -> List[str]:
    payload = _load_yaml(cfg_path)
    tasks = payload.get("daily_tasks") or []
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"daily_tasks is empty in {cfg_path}")
    return [str(t).strip() for t in tasks if str(t).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily T-1 data update")
    parser.add_argument("--config", default="config/daily_tminus1.yaml")
    parser.add_argument("--date", help="YYYYMMDD, default is today-1")
    parser.add_argument("--tasks", help="Comma-separated task names (override config)")
    parser.add_argument("--sleep", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = add_project_root()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_names = _load_task_list(cfg_path)

    tasks_cfg = _load_yaml(project_root / "config" / "tushare_tasks.yaml").get("tasks", {})
    target_dt = _parse_date(args.date)
    output_root = get_tushare_root(ensure=True)

    log(f"tminus1 date: {target_dt.strftime('%Y%m%d')}")
    log(f"tasks: {', '.join(task_names)}")

    for idx, name in enumerate(task_names, 1):
        if name not in tasks_cfg:
            log(f"[{idx}/{len(task_names)}] skip: task not found: {name}")
            continue
        task = TaskConfig.from_dict(name, tasks_cfg[name])
        start_date = _format_date_for_task(task, target_dt)
        end_date = start_date
        log(f"[{idx}/{len(task_names)}] run {name} {start_date}~{end_date}")
        run_task(
            task=task,
            start_date=start_date,
            end_date=end_date,
            output_root=output_root,
            sleep_seconds=args.sleep,
            resume=True,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    sys.exit(main())
