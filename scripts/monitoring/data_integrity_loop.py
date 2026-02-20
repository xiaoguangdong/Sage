#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from scripts.data._shared.runtime import get_log_dir, get_tushare_root
from scripts.data.data_integrity_checker import DataIntegrityChecker


def _default_paths() -> tuple[Path, Path, Path]:
    log_dir = get_log_dir("data")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = log_dir / f"data_integrity_report_{ts}.txt"
    json_path = log_dir / f"data_integrity_report_{ts}.json"
    plan_path = Path("config") / "download_plans.yaml"
    return report_path, json_path, plan_path


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="数据完整性闭环：检查 -> 生成补数计划 -> (可选)执行 -> 复核")
    parser.add_argument("--config", default="config/tushare_tasks.yaml")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--plan-name", default="")
    parser.add_argument("--plan-out", default="")
    parser.add_argument("--report-out", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--execute", action="store_true", help="执行补数计划（调用 batch_download_missing.py）")
    parser.add_argument("--sleep", type=int, default=40, help="补数任务请求间隔秒数")
    parser.add_argument("--full-scan", action="store_true", help="关闭轻量模式，读取全量数据")
    args = parser.parse_args()

    report_path, json_path, default_plan_path = _default_paths()
    report_path = Path(args.report_out) if args.report_out else report_path
    json_path = Path(args.json_out) if args.json_out else json_path
    plan_path = Path(args.plan_out) if args.plan_out else default_plan_path

    data_root = Path(args.data_root) if args.data_root else get_tushare_root()
    checker = DataIntegrityChecker(
        config_path=Path(args.config),
        data_root=data_root,
        start_year=args.start_year,
        end_year=args.end_year,
        light_mode=not args.full_scan,
    )

    results = checker.check_all()
    report = checker.generate_report(results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    plan_name = args.plan_name or f"补充历史数据_{datetime.now().strftime('%Y%m%d')}"
    plan = checker.build_backfill_plan(results, plan_name=plan_name)
    plan_payload = {"download_plans": plan}
    _write_yaml(plan_path, plan_payload)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({"results": results, "plan": plan_payload}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"✅ 完整性报告: {report_path}")
    print(f"✅ 补数计划: {plan_path} (plan={plan_name})")
    print(f"✅ JSON摘要: {json_path}")

    if args.execute:
        print("🚀 执行补数计划...")
        command = [
            sys.executable,
            "scripts/data/batch_download_missing.py",
            "--plan",
            plan_name,
            "--sleep",
            str(args.sleep),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
