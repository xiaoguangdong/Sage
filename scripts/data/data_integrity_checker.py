#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tushare数据完整性校验器

功能：
1. 读取tushare_tasks.yaml配置
2. 检查每个任务的数据文件是否存在
3. 对于时间序列数据，检查2016-2026年的数据完整性
4. 输出缺失数据报告
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml

from scripts.data._shared.runtime import get_log_dir, get_tushare_root


class DataIntegrityChecker:
    """数据完整性校验器"""

    def __init__(
        self,
        config_path: Path,
        data_root: Path,
        start_year: int = 2016,
        end_year: int = 2026,
        light_mode: bool = True,
    ):
        """初始化

        Args:
            config_path: tushare_tasks.yaml配置文件路径
            data_root: 数据根目录
            start_year: 起始年份
            end_year: 结束年份
        """
        self.config_path = config_path
        self.data_root = data_root
        self.start_year = start_year
        self.end_year = end_year
        self.light_mode = light_mode

        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.tasks = self.config.get("tasks", {})
        self.integrity_exclude: Set[str] = set(self.config.get("integrity_exclude", []) or [])

        policy = self.config.get("missing_handling", {}) or {}
        delayed_grace_days = policy.get("delayed_grace_days", 7)
        try:
            delayed_grace_days = int(delayed_grace_days)
        except Exception:
            delayed_grace_days = 7
        delayed_grace_days = max(0, delayed_grace_days)

        structural_tasks = set(policy.get("structural_missing_tasks", []) or [])
        structural_tasks.update(self.integrity_exclude)
        skip_classes = set(policy.get("skip_missing_classes", ["structural_missing"]) or ["structural_missing"])

        self.missing_policy = {
            "structural_missing_tasks": structural_tasks,
            "skip_missing_classes": skip_classes,
            "delayed_grace_days": delayed_grace_days,
            "delayed_grace_by_task": policy.get("delayed_grace_by_task", {}) or {},
        }

    def _find_actual_file(self, expected_path: Path) -> Optional[Path]:
        """查找实际的文件路径（支持多种命名模式）

        Args:
            expected_path: 配置中期望的路径

        Returns:
            实际文件路径，如果不存在返回None
        """
        # 1. 直接检查期望路径
        if expected_path.exists():
            return expected_path

        # 2. 检查 _all 后缀版本
        if expected_path.suffix == ".parquet":
            all_version = expected_path.parent / f"{expected_path.stem}_all.parquet"
            if all_version.exists():
                return all_version

        # 3. 检查同名目录（如 daily.parquet -> daily/ 目录）
        stem = expected_path.stem
        same_name_dir = expected_path.parent / stem
        if same_name_dir.exists() and same_name_dir.is_dir():
            # 检查目录下是否有parquet文件
            parquet_files = list(same_name_dir.glob("*.parquet"))
            if parquet_files:
                return same_name_dir

        # 4. 检查按年份分割的文件（如 daily/daily_2020.parquet）
        if expected_path.parent.exists():
            # 查找同名目录下的年份文件
            parent = expected_path.parent
            year_files = list(parent.glob(f"{stem}_20*.parquet"))
            if year_files:
                return parent

        # 5. 检查父目录下的同名文件
        parent_file = self.data_root / f"{stem}_all.parquet"
        if parent_file.exists():
            return parent_file

        # 6. 检查 data/raw/tushare 目录（旧数据位置）
        raw_path = self.data_root.parent / "raw" / "tushare" / expected_path.relative_to(self.data_root)
        if raw_path.exists():
            return raw_path

        return None

    def check_all(self) -> Dict[str, Dict]:
        """检查所有任务的数据完整性

        Returns:
            检查结果字典
        """
        results = {}

        print(f"开始检查数据完整性 ({self.start_year}-{self.end_year})")
        print("=" * 80)

        for task_name, task_config in self.tasks.items():
            if task_name in self.integrity_exclude:
                print(f"\n跳过任务: {task_name} (integrity_exclude)")
                results[task_name] = {
                    "task_name": task_name,
                    "mode": task_config.get("mode", "single"),
                    "output_path": str(self.data_root / task_config["output"]),
                    "file_exists": False,
                    "missing_data": [],
                    "status": "skipped",
                    "missing_class": "structural_missing",
                    "skip_backfill": True,
                    "skip_reason": "integrity_exclude",
                }
                continue
            print(f"\n检查任务: {task_name}")
            result = self._check_task(task_name, task_config)
            result = self._attach_missing_meta(task_name, result, task_config)
            results[task_name] = result

        return results

    def _task_delayed_grace_days(self, task_name: str) -> int:
        override = (self.missing_policy.get("delayed_grace_by_task") or {}).get(task_name)
        if override is None:
            return int(self.missing_policy.get("delayed_grace_days", 7))
        try:
            return max(0, int(override))
        except Exception:
            return int(self.missing_policy.get("delayed_grace_days", 7))

    def _classify_missing(
        self, task_name: str, result: Dict[str, Any], task_config: Dict[str, Any]
    ) -> Tuple[str, bool, str]:
        status = result.get("status")
        structural_tasks: Set[str] = set(self.missing_policy.get("structural_missing_tasks") or set())
        skip_classes: Set[str] = set(self.missing_policy.get("skip_missing_classes") or set())

        if status in {"ok"}:
            return "none", False, ""
        if status == "skipped":
            return "structural_missing", True, result.get("skip_reason", "integrity_exclude")
        if task_name in structural_tasks:
            cls = "structural_missing"
            return cls, cls in skip_classes, "任务在结构性缺失名单（数据源无/权限未开）"

        if status == "incomplete":
            max_date = self._parse_date(result.get("max_date"))
            if max_date is not None:
                target_end = self._target_window()[1]
                lag_days = (target_end - max_date).days
                grace_days = self._task_delayed_grace_days(task_name)
                if 0 <= lag_days <= grace_days:
                    cls = "delayed"
                    return cls, cls in skip_classes, f"数据发布延迟窗口内（滞后 {lag_days} 天，阈值 {grace_days} 天）"

        cls = "error"
        return cls, cls in skip_classes, "需要补数或排查抓取/写入错误"

    def _attach_missing_meta(
        self, task_name: str, result: Dict[str, Any], task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        missing_class, skip_backfill, skip_reason = self._classify_missing(task_name, result, task_config)
        result["missing_class"] = missing_class
        result["skip_backfill"] = bool(skip_backfill)
        result["skip_reason"] = skip_reason
        return result

    def _read_parquet(
        self,
        path: Path,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if not self.light_mode or not columns:
            return pd.read_parquet(path)
        try:
            return pd.read_parquet(path, columns=columns)
        except Exception:
            return pd.read_parquet(path)

    def _check_task(self, task_name: str, task_config: Dict) -> Dict:
        """检查单个任务的数据完整性

        Args:
            task_name: 任务名称
            task_config: 任务配置

        Returns:
            检查结果
        """
        output_path = self.data_root / task_config["output"]
        mode = task_config.get("mode", "single")

        result = {
            "task_name": task_name,
            "mode": mode,
            "output_path": str(output_path),
            "file_exists": output_path.exists(),
            "missing_data": [],
            "status": "unknown",
        }

        # 1. 检查文件是否存在（支持多种命名模式）
        actual_path = self._find_actual_file(output_path)

        if actual_path is None:
            result["status"] = "missing_file"
            print(f"  ❌ 文件不存在: {output_path}")
            return result

        # 更新为实际路径
        output_path = actual_path
        result["actual_path"] = str(actual_path)
        result["file_exists"] = True

        # 2. 读取数据（合并所有能找到的数据源）
        try:
            dfs = []
            sources = []

            dedup_keys = task_config.get("dedup_keys") or []
            candidate_cols = [
                "trade_date",
                "ann_date",
                "end_date",
                "cal_date",
                "period",
                "f_ann_date",
            ]
            columns = list({*dedup_keys, *candidate_cols})

            if actual_path.is_dir():
                # 读取目录下所有parquet文件
                parquet_files = list(actual_path.glob("*.parquet"))
                if not parquet_files:
                    result["status"] = "empty_directory"
                    print(f"  ❌ 目录为空: {actual_path}")
                    return result
                for pf in parquet_files:
                    dfs.append(self._read_parquet(pf, columns=columns))
                sources.append(f"目录({len(parquet_files)}个文件)")
            else:
                dfs.append(self._read_parquet(actual_path, columns=columns))
                sources.append(actual_path.name)

            # 同时查找 _all 版本和分片目录，合并更完整的数据
            stem = Path(task_config["output"]).stem
            parent = self.data_root / Path(task_config["output"]).parent

            # 查找 _all 文件
            all_file = parent / f"{stem}_all.parquet"
            if all_file.exists() and str(all_file) != str(actual_path):
                dfs.append(self._read_parquet(all_file, columns=columns))
                sources.append(f"{all_file.name}")

            # 查找同名分片目录
            split_dir = parent / stem
            if split_dir.exists() and split_dir.is_dir() and str(split_dir) != str(actual_path):
                split_files = list(split_dir.glob("*.parquet"))
                if split_files:
                    for sf in split_files:
                        dfs.append(self._read_parquet(sf, columns=columns))
                    sources.append(f"{stem}/({len(split_files)}个分片)")

            # 合并去重
            df = pd.concat(dfs, ignore_index=True)
            if dedup_keys and all(k in df.columns for k in dedup_keys):
                df = df.drop_duplicates(subset=dedup_keys, keep="last")

            result["record_count"] = len(df)
            result["sources"] = sources
            print(f"  ✅ 数据来源: {', '.join(sources)}，记录数: {len(df):,}")
        except Exception as e:
            result["status"] = "read_error"
            result["error"] = str(e)
            print(f"  ❌ 读取失败: {e}")
            return result

        # 3. 根据模式检查数据完整性
        if mode == "single":
            # 单次下载任务，只检查是否有数据
            result["status"] = "ok" if len(df) > 0 else "empty"

        elif mode == "date_range":
            # 日期范围任务，检查时间覆盖
            result = self._check_date_range(df, task_config, result)

        elif mode == "year_quarters":
            # 季度任务，检查季度覆盖
            result = self._check_year_quarters(df, task_config, result)

        elif mode == "list":
            # 列表任务，检查时间覆盖（如果有时间字段）
            result = self._check_list_mode(df, task_config, result)

        return result

    def _check_date_range(self, df: pd.DataFrame, task_config: Dict, result: Dict) -> Dict:
        """检查日期范围任务的数据完整性

        检查三个维度：
        1. 起始日期是否覆盖目标起始
        2. 结束日期是否覆盖当前日期（而非遥远的未来）
        3. 中间是否有年份空洞
        """
        # 查找日期字段
        date_field = None
        for field in ["trade_date", "ann_date", "end_date"]:
            if field in df.columns:
                date_field = field
                break

        if date_field is None:
            result["status"] = "no_date_field"
            print("  ⚠️  未找到日期字段")
            return result

        # 转换日期格式
        df[date_field] = pd.to_datetime(df[date_field], format="%Y%m%d", errors="coerce")

        # 获取数据时间范围
        min_date = df[date_field].min()
        max_date = df[date_field].max()

        result["min_date"] = min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None
        result["max_date"] = max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None

        print(f"  📅 时间范围: {result['min_date']} ~ {result['max_date']}")

        if pd.isna(min_date) or pd.isna(max_date):
            result["status"] = "invalid_dates"
            return result

        # 用当前日期作为结束目标（而非 end_year 年底，未来数据不可能有）
        target_start = datetime(self.start_year, 1, 1)
        now = datetime.now()
        target_end = min(datetime(self.end_year, 12, 31), now - timedelta(days=7))

        issues = []

        # 1. 检查起始日期
        if min_date > target_start + timedelta(days=30):
            issues.append(f"缺少早期数据: {target_start.year}年初 ~ {min_date.strftime('%Y-%m-%d')}")

        # 2. 检查结束日期
        if max_date < target_end:
            issues.append(f"缺少近期数据: {max_date.strftime('%Y-%m-%d')} ~ {target_end.strftime('%Y-%m-%d')}")

        # 3. 检查中间年份空洞
        years_with_data = set(df[date_field].dt.year.dropna().unique())
        expected_years = set(range(max(self.start_year, min_date.year), min(self.end_year, max_date.year) + 1))
        missing_years = sorted(expected_years - years_with_data)
        if missing_years:
            issues.append(f"中间年份空洞: {', '.join(str(y) for y in missing_years)}")
            result["missing_years"] = missing_years

        # 4. 按年统计记录数
        year_counts = df[date_field].dt.year.value_counts().sort_index()
        result["year_counts"] = {int(y): int(c) for y, c in year_counts.items()}

        if issues:
            result["status"] = "incomplete"
            result["missing_data"] = issues
            for issue in issues:
                print(f"  ⚠️  {issue}")
        else:
            result["status"] = "ok"
            print("  ✅ 数据完整")

        return result

    def _parse_date(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except Exception:
            return None

    def _format_date(self, value: datetime) -> str:
        return value.strftime("%Y%m%d")

    def _target_window(self) -> Tuple[datetime, datetime]:
        target_start = datetime(self.start_year, 1, 1)
        now = datetime.now()
        target_end = min(datetime(self.end_year, 12, 31), now - timedelta(days=7))
        return target_start, target_end

    def _infer_missing_ranges(self, result: Dict) -> List[Tuple[str, str, str]]:
        ranges: List[Tuple[str, str, str]] = []
        target_start, target_end = self._target_window()
        min_date = self._parse_date(result.get("min_date"))
        max_date = self._parse_date(result.get("max_date"))
        missing_years = result.get("missing_years") or []

        if min_date and min_date > target_start + timedelta(days=30):
            ranges.append(
                (self._format_date(target_start), self._format_date(min_date - timedelta(days=1)), "缺少早期数据")
            )
        if max_date and max_date < target_end - timedelta(days=1):
            ranges.append(
                (self._format_date(max_date + timedelta(days=1)), self._format_date(target_end), "缺少近期数据")
            )
        for year in missing_years:
            ranges.append((f"{year}0101", f"{year}1231", f"缺少年份 {year}"))

        if not ranges:
            ranges.append((self._format_date(target_start), self._format_date(target_end), "补齐缺口/全量回补"))
        return ranges

    def build_backfill_plan(self, results: Dict[str, Dict], plan_name: Optional[str] = None) -> Dict[str, List[Dict]]:
        plan_items: List[Dict] = []
        plan_name = plan_name or f"补充历史数据_{datetime.now().strftime('%Y%m%d')}"

        for task_name, result in results.items():
            status = result.get("status")
            if status not in {"incomplete", "missing_file", "empty", "invalid_dates"}:
                continue
            if result.get("skip_backfill"):
                print(
                    f"跳过补数任务: {task_name} "
                    f"(missing_class={result.get('missing_class')}, reason={result.get('skip_reason')})"
                )
                continue
            task_config = self.tasks.get(task_name, {})
            mode = result.get("mode") or task_config.get("mode", "single")
            missing_class = result.get("missing_class", "error")

            if mode in {"date_range", "list"}:
                ranges = self._infer_missing_ranges(result)
                for start_date, end_date, reason in ranges:
                    plan_items.append(
                        {
                            "task": task_name,
                            "desc": f"{task_name} {reason}",
                            "start_date": start_date,
                            "end_date": end_date,
                            "missing_class": missing_class,
                        }
                    )
            elif mode == "year_quarters":
                missing_periods = result.get("missing_periods") or []
                years = sorted({p[:4] for p in missing_periods if len(p) >= 4})
                for year in years:
                    plan_items.append(
                        {
                            "task": task_name,
                            "desc": f"{task_name} 缺失季度 {year}",
                            "start_date": f"{year}0101",
                            "end_date": f"{year}1231",
                            "missing_class": missing_class,
                        }
                    )

        return {plan_name: plan_items}

    def _check_year_quarters(self, df: pd.DataFrame, task_config: Dict, result: Dict) -> Dict:
        """检查季度任务的数据完整性

        Args:
            df: 数据DataFrame
            task_config: 任务配置
            result: 当前结果

        Returns:
            更新后的结果
        """
        # 查找季度字段
        period_field = None
        for field in ["end_date", "period", "f_ann_date"]:
            if field in df.columns:
                period_field = field
                break

        if period_field is None:
            result["status"] = "no_period_field"
            print("  ⚠️  未找到季度字段")
            return result

        # 提取已有的季度
        df["period_str"] = df[period_field].astype(str).str[:8]
        existing_periods = set(df["period_str"].unique())

        # 生成目标季度列表
        start_year = task_config.get("start_year", self.start_year)
        end_year = task_config.get("end_year", self.end_year)
        quarters = task_config.get("quarters", ["0331", "0630", "0930", "1231"])

        target_periods = []
        for year in range(max(start_year, self.start_year), min(end_year, self.end_year) + 1):
            for quarter in quarters:
                target_periods.append(f"{year}{quarter}")

        # 检查缺失的季度
        missing_periods = [p for p in target_periods if p not in existing_periods]

        result["target_periods"] = len(target_periods)
        result["existing_periods"] = len(existing_periods)
        result["missing_periods"] = missing_periods

        if missing_periods:
            result["status"] = "incomplete"
            result["missing_data"] = missing_periods
            print(f"  ⚠️  缺失季度: {len(missing_periods)}/{len(target_periods)}")
            print(f"     {', '.join(missing_periods[:5])}{'...' if len(missing_periods) > 5 else ''}")
        else:
            result["status"] = "ok"
            print(f"  ✅ 季度完整: {len(target_periods)}/{len(target_periods)}")

        return result

    def _check_list_mode(self, df: pd.DataFrame, task_config: Dict, result: Dict) -> Dict:
        """检查列表模式任务的数据完整性

        Args:
            df: 数据DataFrame
            task_config: 任务配置
            result: 当前结果

        Returns:
            更新后的结果
        """
        # 列表模式任务，如果有时间字段则检查时间覆盖
        if "start_field" in task_config and "end_field" in task_config:
            return self._check_date_range(df, task_config, result)
        else:
            # 没有时间字段，只检查是否有数据
            result["status"] = "ok" if len(df) > 0 else "empty"
            print("  ✅ 数据存在")
            return result

    def generate_report(self, results: Dict[str, Dict]) -> str:
        """生成检查报告

        Args:
            results: 检查结果

        Returns:
            报告文本
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Tushare数据完整性检查报告")
        report_lines.append(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"目标年份: {self.start_year}-{self.end_year}")
        report_lines.append("=" * 80)

        # 统计
        total = len(results)
        ok_count = sum(1 for r in results.values() if r["status"] == "ok")
        incomplete_count = sum(1 for r in results.values() if r["status"] == "incomplete")
        missing_count = sum(1 for r in results.values() if r["status"] == "missing_file")
        error_count = sum(1 for r in results.values() if r["status"] in ["read_error", "invalid_dates"])

        report_lines.append(f"\n总任务数: {total}")
        report_lines.append(f"  ✅ 完整: {ok_count}")
        report_lines.append(f"  ⚠️  不完整: {incomplete_count}")
        report_lines.append(f"  ❌ 文件缺失: {missing_count}")
        report_lines.append(f"  ❌ 读取错误: {error_count}")

        class_counts = defaultdict(int)
        for r in results.values():
            class_counts[r.get("missing_class", "none")] += 1
        report_lines.append(
            "  分类统计: "
            f"none={class_counts['none']}, "
            f"structural_missing={class_counts['structural_missing']}, "
            f"delayed={class_counts['delayed']}, "
            f"error={class_counts['error']}"
        )

        # 详细信息
        report_lines.append("\n" + "=" * 80)
        report_lines.append("详细信息")
        report_lines.append("=" * 80)

        # 按状态分组
        status_groups = defaultdict(list)
        for task_name, result in results.items():
            status_groups[result["status"]].append((task_name, result))

        # 1. 文件缺失
        if status_groups["missing_file"]:
            report_lines.append("\n【文件缺失】")
            for task_name, result in status_groups["missing_file"]:
                report_lines.append(f"  - {task_name}: {result['output_path']}")

        # 2. 数据不完整
        if status_groups["incomplete"]:
            report_lines.append("\n【数据不完整】")
            for task_name, result in status_groups["incomplete"]:
                report_lines.append(f"\n  {task_name}:")
                if result.get("min_date") and result.get("max_date"):
                    report_lines.append(
                        f"    实际范围: {result['min_date']} ~ {result['max_date']}  ({result.get('record_count', '?'):,} 条)"
                    )
                if result.get("year_counts"):
                    years_str = ", ".join(f"{y}:{c:,}" for y, c in sorted(result["year_counts"].items()))
                    report_lines.append(f"    按年分布: {years_str}")
                if result.get("missing_periods"):
                    report_lines.append(f"    缺失季度: {len(result['missing_periods'])}个")
                    report_lines.append(f"    {', '.join(result['missing_periods'][:10])}")
                    if len(result["missing_periods"]) > 10:
                        report_lines.append(f"    ... 还有 {len(result['missing_periods']) - 10} 个")
                elif result.get("missing_data"):
                    for missing in result["missing_data"]:
                        report_lines.append(f"    - {missing}")
                if result.get("missing_class") and result.get("missing_class") != "none":
                    report_lines.append(
                        f"    分类: {result.get('missing_class')} "
                        f"(skip_backfill={result.get('skip_backfill')}, reason={result.get('skip_reason')})"
                    )

        # 3. 读取错误
        if status_groups["read_error"] or status_groups["invalid_dates"]:
            report_lines.append("\n【读取错误】")
            for task_name, result in list(status_groups["read_error"]) + list(status_groups["invalid_dates"]):
                report_lines.append(f"  - {task_name}: {result.get('error', result['status'])}")

        # 4. 完整数据（简要列出）
        if status_groups["ok"]:
            report_lines.append(f"\n【数据完整】({len(status_groups['ok'])}个任务)")
            for task_name, result in status_groups["ok"][:5]:
                report_lines.append(f"  ✅ {task_name}")
            if len(status_groups["ok"]) > 5:
                report_lines.append(f"  ... 还有 {len(status_groups['ok']) - 5} 个任务")

        report_lines.append("\n" + "=" * 80)

        return "\n".join(report_lines)


def main():
    """主函数"""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Tushare数据完整性校验")
    parser.add_argument("--config", default=str(ROOT / "config/tushare_tasks.yaml"))
    parser.add_argument("--data-root", default="")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--report-out", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--plan-out", default="")
    parser.add_argument("--plan-name", default="")
    parser.add_argument("--full-scan", action="store_true", help="关闭轻量模式，读取全量数据")
    args = parser.parse_args()

    config_path = Path(args.config)
    data_root = Path(args.data_root) if args.data_root else get_tushare_root()

    checker = DataIntegrityChecker(
        config_path=config_path,
        data_root=data_root,
        start_year=args.start_year,
        end_year=args.end_year,
        light_mode=not args.full_scan,
    )

    # 执行检查
    results = checker.check_all()

    # 生成报告
    report = checker.generate_report(results)
    print("\n" + report)

    # 保存报告
    report_path = Path(args.report_out) if args.report_out else get_log_dir("data") / "data_integrity_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"\n报告已保存: {report_path}")

    # 生成补数计划
    plan_payload = None
    if args.plan_out:
        plan = checker.build_backfill_plan(results, plan_name=args.plan_name or None)
        plan_payload = {"download_plans": plan}
        plan_path = Path(args.plan_out)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with plan_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(plan_payload, f, allow_unicode=True, sort_keys=False)
        print(f"补数计划已保存: {plan_path}")

    if args.json_out:
        payload = {"results": results, "plan": plan_payload}
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
