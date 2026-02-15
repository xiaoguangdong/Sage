#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MODULE_RULES = {
    "fetch_": "data",
    "concept_": "strategy",
    "calculate_": "models",
    "training": "models",
    "refetch_": "data",
    "backtest": "backtest",
}


def infer_module(filename: str) -> str:
    lower = filename.lower()
    for prefix, module in MODULE_RULES.items():
        if lower.startswith(prefix):
            return module
    return "misc"


def resolve_target_path(log_root: Path, module: str, name: str) -> Path:
    target_dir = log_root / module
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    index = 1
    while True:
        candidate = target_dir / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def organize(log_root: Path, dry_run: bool = False) -> list[tuple[Path, Path]]:
    moved: list[tuple[Path, Path]] = []
    for src in sorted(log_root.glob("*.log")):
        module = infer_module(src.name)
        dst = resolve_target_path(log_root, module, src.name)
        moved.append((src, dst))
        if not dry_run:
            src.rename(dst)
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 logs 根目录散落日志到 logs/<module>/")
    parser.add_argument("--log-root", default="logs", help="日志根目录")
    parser.add_argument("--dry-run", action="store_true", help="仅输出移动计划")
    args = parser.parse_args()

    root = Path(args.log_root)
    if not root.exists():
        raise FileNotFoundError(f"日志目录不存在: {root}")

    moved = organize(root, dry_run=args.dry_run)
    for src, dst in moved:
        print(f"{src} -> {dst}")
    print(f"总计: {len(moved)} 个文件")


if __name__ == "__main__":
    main()
