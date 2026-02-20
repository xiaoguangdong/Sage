#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

JOB_SCRIPTS = {
    "weekly_pipeline": PROJECT_ROOT / "sage_app" / "pipelines" / "run_weekly.py",
    "stock_monthly": PROJECT_ROOT / "scripts" / "stock" / "run_stock_selector_monthly.py",
    "stock_weekly": PROJECT_ROOT / "scripts" / "stock" / "run_stock_selector_weekly_signal.py",
    "stock_benchmark": PROJECT_ROOT / "scripts" / "stock" / "run_stock_model_benchmark.py",
    "stock_governance": PROJECT_ROOT / "scripts" / "stock" / "run_stock_strategy_governance.py",
    "stock_walk_forward": PROJECT_ROOT / "scripts" / "stock" / "run_stock_walk_forward.py",
    "broker_submit": PROJECT_ROOT / "scripts" / "stock" / "run_broker_execution.py",
    "trend_scheduler": PROJECT_ROOT / "sage_app" / "pipelines" / "trend_signal_scheduler.py",
    "stock_scheduler": PROJECT_ROOT / "sage_app" / "pipelines" / "stock_selector_scheduler.py",
    "macro_prediction": PROJECT_ROOT / "scripts" / "models" / "macro" / "run_macro_prediction.py",
    "industry_concept_bias": PROJECT_ROOT / "scripts" / "strategy" / "build_industry_concept_bias.py",
    "industry_signal_contract": PROJECT_ROOT / "scripts" / "strategy" / "build_industry_signal_contract.py",
    "industry_signal_quality": PROJECT_ROOT / "scripts" / "monitoring" / "check_industry_signal_quality.py",
    "data_integrity_loop": PROJECT_ROOT / "scripts" / "monitoring" / "data_integrity_loop.py",
    "ths_daily_completeness": PROJECT_ROOT / "scripts" / "monitoring" / "check_ths_daily_completeness.py",
    "ths_daily_monthly_full": PROJECT_ROOT / "scripts" / "data" / "run_ths_daily_monthly_full.py",
    "champion_challenger_backtest": PROJECT_ROOT / "scripts" / "backtest" / "backtest_champion_challenger.py",
}


def run_job(job: str, job_args: list[str]) -> int:
    script = JOB_SCRIPTS[job]
    if not script.exists():
        raise FileNotFoundError(f"任务脚本不存在: {script}")
    passthrough = list(job_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    command = [sys.executable, str(script), *passthrough]
    print("执行任务:", " ".join(command))
    result = subprocess.run(command, cwd=str(PROJECT_ROOT))
    return int(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="统一任务入口")
    parser.add_argument("job", choices=sorted(JOB_SCRIPTS.keys()), help="任务名称")
    parser.add_argument("job_args", nargs=argparse.REMAINDER, help="透传给任务脚本的参数")
    args = parser.parse_args()
    raise SystemExit(run_job(args.job, args.job_args))


if __name__ == "__main__":
    main()
