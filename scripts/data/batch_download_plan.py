#!/usr/bin/env python
"""
Tushare 批量下载脚本 - 基于下载计划
支持配置文件和命令行参数两种方式
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from scripts.data.data_integrity_checker import DataIntegrityChecker
from scripts.data.tushare_downloader import TaskConfig, _load_yaml, run_task

DATE_CANDIDATES = [
    "trade_date",
    "ann_date",
    "end_date",
    "cal_date",
    "month",
    "period",
    "f_ann_date",
]
DEFAULT_COVERAGE_GRACE_DAYS = 7


def _coverage_grace_days(date_format: str) -> int:
    if date_format == "%Y%m":
        return 31
    return DEFAULT_COVERAGE_GRACE_DAYS


def _resolve_output(path_str: str, output_root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        return output_root / path
    return path


def _parse_date(value: str, date_format: str) -> Optional[datetime]:
    if value is None:
        return None
    try:
        return datetime.strptime(str(value), date_format)
    except Exception:
        try:
            return pd.to_datetime(value, errors="coerce")
        except Exception:
            return None


def _load_date_bounds(path: Path, date_format: str) -> tuple[Optional[datetime], Optional[datetime]]:
    files: List[Path]
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
    else:
        files = [path]
    min_date = None
    max_date = None
    for file in files:
        try:
            df = pd.read_parquet(file, columns=DATE_CANDIDATES)
        except Exception:
            try:
                df = pd.read_parquet(file)
            except Exception:
                continue
        date_col = None
        for col in DATE_CANDIDATES:
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            continue
        series = df[date_col]
        if series.empty:
            continue
        dt = pd.to_datetime(series.astype(str), format=date_format, errors="coerce")
        if dt.empty:
            continue
        file_min = dt.min()
        file_max = dt.max()
        if pd.isna(file_min) or pd.isna(file_max):
            continue
        if min_date is None or file_min < min_date:
            min_date = file_min
        if max_date is None or file_max > max_date:
            max_date = file_max
    return min_date, max_date


def _should_skip_task(
    task: TaskConfig,
    start_date: Optional[str],
    end_date: Optional[str],
    output_root: Path,
    checker: DataIntegrityChecker,
) -> bool:
    if not start_date or not end_date:
        return False
    # list 模式通常按 ts_code/列表项拉取，仅靠全表最小最大日期无法判断是否完整
    if task.mode != "date_range":
        return False
    output_path = _resolve_output(task.output, output_root)
    actual_path = checker._find_actual_file(output_path)
    if actual_path is None:
        return False
    min_date, max_date = _load_date_bounds(actual_path, task.date_format)
    if min_date is None or max_date is None:
        return False
    plan_start = _parse_date(start_date, task.date_format)
    plan_end = _parse_date(end_date, task.date_format)
    if plan_start is None or plan_end is None or pd.isna(plan_start) or pd.isna(plan_end):
        return False
    grace = timedelta(days=_coverage_grace_days(task.date_format))
    return min_date <= plan_start + grace and max_date >= plan_end - grace


def _build_missing_skip_rules(raw_config: dict) -> dict:
    policy = raw_config.get("missing_handling", {}) or {}
    structural_tasks = set(raw_config.get("integrity_exclude", []) or [])
    structural_tasks.update(policy.get("structural_missing_tasks", []) or [])
    skip_classes = set(policy.get("skip_missing_classes", ["structural_missing"]) or ["structural_missing"])
    return {
        "structural_tasks": structural_tasks,
        "skip_classes": skip_classes,
    }


def _should_skip_by_missing_rule(plan: dict, task_name: str, skip_rules: dict) -> tuple[bool, str]:
    if bool(plan.get("force_run")):
        return False, ""

    if bool(plan.get("skip_backfill")):
        reason = str(plan.get("skip_reason") or "计划标记为 skip_backfill")
        return True, reason

    missing_class = str(plan.get("missing_class") or "").strip()
    if missing_class and missing_class in skip_rules["skip_classes"]:
        return True, f"missing_class={missing_class} 命中跳过规则"

    if task_name in skip_rules["structural_tasks"]:
        return True, "任务在结构性缺失名单（数据源无/权限未开）"

    return False, ""


def log(msg: str):
    """带时间戳的日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def load_download_plan(plan_name: str = None):
    """从配置文件加载下载计划"""
    plan_path = project_root / "config" / "download_plans_20260220.yaml"
    plan_config = _load_yaml(plan_path)

    if plan_name is None:
        # 使用最新的计划
        plan_name = "补充历史数据_20260220"

    if plan_name not in plan_config["download_plans"]:
        raise ValueError(f"计划 {plan_name} 不存在")

    return plan_name, plan_config["download_plans"][plan_name]


def main():
    """批量下载缺失数据"""

    parser = argparse.ArgumentParser(description="Tushare 批量下载脚本")
    parser.add_argument("--plan", type=str, help="下载计划名称（从 download_plans_20260220.yaml 读取）")
    parser.add_argument("--tasks", type=str, help="任务列表，逗号分隔（如: sw_valuation,forecast）")
    parser.add_argument("--start-date", type=str, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", type=str, help="结束日期 YYYYMMDD")
    parser.add_argument("--sleep", type=int, default=40, help="请求间隔秒数（默认40）")

    args = parser.parse_args()

    # 加载任务配置
    config_path = project_root / "config" / "tushare_tasks.yaml"
    raw_config = _load_yaml(config_path)

    # 获取 tasks 字典
    tasks_dict = raw_config.get("tasks", raw_config)
    data_root = project_root / "data" / "tushare"
    checker = DataIntegrityChecker(config_path, data_root, light_mode=True)
    missing_skip_rules = _build_missing_skip_rules(raw_config)

    # 将配置转换为 TaskConfig 对象
    all_tasks = {}
    for task_name, task_dict in tasks_dict.items():
        # 跳过空配置或缺少必需字段的任务
        if not task_dict or "api" not in task_dict or "output" not in task_dict:
            continue
        all_tasks[task_name] = TaskConfig.from_dict(task_name, task_dict)

    # 确定下载计划
    if args.tasks:
        # 命令行参数模式
        task_names = [t.strip() for t in args.tasks.split(",")]
        if not args.start_date or not args.end_date:
            log("❌ 使用 --tasks 时必须指定 --start-date 和 --end-date")
            return

        download_plan = [
            {
                "task": name,
                "desc": f"{name} {args.start_date}-{args.end_date}",
                "start_date": args.start_date,
                "end_date": args.end_date,
            }
            for name in task_names
        ]
        plan_name = "命令行参数"
    else:
        # 配置文件模式
        plan_name, download_plan = load_download_plan(args.plan)

    total = len(download_plan)
    log(f"开始批量下载，计划: {plan_name}")
    log(f"共 {total} 个任务，请求间隔: {args.sleep}秒")
    log("=" * 60)

    start_time = time.time()
    success_count = 0
    failed_tasks = []

    for idx, plan in enumerate(download_plan, 1):
        task_name = plan["task"]
        desc = plan["desc"]
        start_date = plan["start_date"]
        end_date = plan["end_date"]

        log("")
        log("=" * 60)
        log(f"[{idx}/{total}] {task_name} - {desc}")
        log(f"参数: start_date={start_date}, end_date={end_date}")
        log("=" * 60)

        task_start = time.time()

        try:
            # 从配置中获取任务
            if task_name not in all_tasks:
                log(f"❌ 任务 {task_name} 不在配置文件中")
                failed_tasks.append(task_name)
                continue

            task = all_tasks[task_name]

            skip_by_rule, skip_reason = _should_skip_by_missing_rule(plan, task_name, missing_skip_rules)
            if skip_by_rule:
                log(f"⏭️  跳过 {task_name}：{skip_reason}")
                success_count += 1
                continue

            if _should_skip_task(task, start_date, end_date, data_root, checker):
                log(f"⏭️  跳过 {task_name}：已有数据覆盖 {start_date}~{end_date}")
                success_count += 1
                continue

            # 执行下载
            run_task(
                task=task,
                start_date=start_date,
                end_date=end_date,
                output_root=project_root / "data" / "tushare",
                sleep_seconds=args.sleep,
                resume=True,
                dry_run=False,
            )

            task_elapsed = time.time() - task_start
            log(f"✅ {task_name} 完成，耗时: {task_elapsed/60:.1f} 分钟")
            success_count += 1

        except KeyboardInterrupt:
            log("⚠️  用户中断")
            break
        except Exception as e:
            task_elapsed = time.time() - task_start
            log(f"❌ {task_name} 失败: {e}")
            log(f"   耗时: {task_elapsed/60:.1f} 分钟")
            failed_tasks.append(task_name)
            continue

    # 总结
    total_elapsed = time.time() - start_time
    log("")
    log("=" * 60)
    log("批量下载完成")
    log(f"总耗时: {total_elapsed/3600:.1f} 小时")
    log(f"成功: {success_count}/{total}")
    if failed_tasks:
        log(f"失败: {', '.join(failed_tasks)}")
    log("=" * 60)


if __name__ == "__main__":
    main()
