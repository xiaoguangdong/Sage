"""
简易DAG调度器：按配置编排任务执行（顺序+依赖）

示例:
  python sage_app/pipelines/dag_runner.py --pipeline weekly_default
  python sage_app/pipelines/dag_runner.py --pipeline weekly_default --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_JOB = PROJECT_ROOT / "scripts" / "run_job.py"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "app" / "pipeline_dag.yaml"

sys.path.append(str(PROJECT_ROOT))
from scripts.data._shared.runtime import setup_logger  # noqa: E402

logger = setup_logger("dag_runner", module="jobs")


@dataclass
class TaskResult:
    task_id: str
    status: str
    returncode: int
    started_at: datetime
    ended_at: datetime
    message: str | None = None


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"DAG配置不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _topo_sort(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    task_map = {t["id"]: t for t in tasks}
    deps = {t["id"]: set(t.get("depends_on", [])) for t in tasks}
    ready = [tid for tid, ds in deps.items() if not ds]
    ordered: List[str] = []

    while ready:
        tid = ready.pop(0)
        ordered.append(tid)
        for other, ds in deps.items():
            if tid in ds:
                ds.remove(tid)
                if not ds and other not in ordered and other not in ready:
                    ready.append(other)

    if len(ordered) != len(tasks):
        missing = set(task_map.keys()) - set(ordered)
        raise ValueError(f"DAG存在循环或依赖缺失: {missing}")

    return [task_map[tid] for tid in ordered]


def _build_command(task: Dict[str, Any]) -> List[str]:
    if "job" in task:
        job = task["job"]
        args = task.get("args", []) or []
        return [sys.executable, str(RUN_JOB), job, "--", *args]
    if "cmd" in task:
        cmd = task["cmd"]
        if isinstance(cmd, list):
            return cmd
        return [str(cmd)]
    raise ValueError(f"任务缺少 job/cmd: {task.get('id')}")


def _run_command(cmd: List[str], env: Dict[str, str] | None = None) -> int:
    logger.info("执行: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return int(result.returncode)


def _try_db_logger() -> Any | None:
    try:
        import psycopg  # type: ignore

        host = os.getenv("SAGE_DB_HOST", "127.0.0.1")
        port = os.getenv("SAGE_DB_PORT", "5432")
        name = os.getenv("SAGE_DB_NAME", "sage_db")
        user = os.getenv("SAGE_DB_USER", "sage")
        password = os.getenv("SAGE_DB_PASSWORD", "sage_dev_2026")
        dsn = f"postgresql://{user}:{password}@{host}:{port}/{name}"
        return psycopg.connect(dsn)
    except Exception as exc:  # pragma: no cover - 可选能力
        logger.warning("跳过DB日志: %s", exc)
        return None


def _log_job_start(conn: Any | None, job_name: str) -> int | None:
    if conn is None:
        return None
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO meta.job_run_log (job_name, status, started_at)
            VALUES (%s, %s, NOW())
            RETURNING id
            """,
            (job_name, "running"),
        )
        job_id = cur.fetchone()[0]
    conn.commit()
    return job_id


def _log_job_end(conn: Any | None, job_id: int | None, status: str, message: str | None = None) -> None:
    if conn is None or job_id is None:
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE meta.job_run_log
            SET status=%s, ended_at=NOW(), message=%s
            WHERE id=%s
            """,
            (status, message, job_id),
        )
    conn.commit()


def run_pipeline(config: Dict[str, Any], pipeline_name: str, dry_run: bool = False) -> List[TaskResult]:
    pipelines = config.get("pipelines") or {}
    if pipeline_name not in pipelines:
        raise KeyError(f"未找到pipeline: {pipeline_name}")

    pipeline = pipelines[pipeline_name] or {}
    tasks = pipeline.get("tasks") or []
    if not tasks:
        raise ValueError(f"pipeline为空: {pipeline_name}")

    ordered = _topo_sort(tasks)
    stop_on_fail = bool(pipeline.get("stop_on_fail", True))

    conn = _try_db_logger()
    results: List[TaskResult] = []

    for task in ordered:
        if not task.get("enabled", True):
            logger.info("跳过任务: %s (disabled)", task.get("id"))
            continue

        task_id = task["id"]
        job_name = f"{pipeline_name}.{task_id}"
        job_id = _log_job_start(conn, job_name)
        started_at = datetime.now()

        if dry_run:
            cmd = _build_command(task)
            logger.info("[dry-run] %s -> %s", task_id, " ".join(cmd))
            status = "skipped"
            returncode = 0
        else:
            cmd = _build_command(task)
            env = os.environ.copy()
            env.update(task.get("env", {}) or {})
            returncode = _run_command(cmd, env=env)
            status = "success" if returncode == 0 else "failed"

        ended_at = datetime.now()
        _log_job_end(conn, job_id, status, None if returncode == 0 else f"returncode={returncode}")
        results.append(
            TaskResult(
                task_id=task_id,
                status=status,
                returncode=returncode,
                started_at=started_at,
                ended_at=ended_at,
            )
        )

        if returncode != 0 and stop_on_fail:
            logger.error("任务失败，终止pipeline: %s", task_id)
            break

    if conn:
        conn.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DAG任务编排器")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="DAG配置路径")
    parser.add_argument("--pipeline", type=str, required=True, help="pipeline名称")
    parser.add_argument("--dry-run", action="store_true", help="仅打印，不执行")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    results = run_pipeline(config, args.pipeline, dry_run=args.dry_run)
    failed = [r for r in results if r.returncode != 0]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
