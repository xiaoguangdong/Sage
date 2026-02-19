#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用历史数据补充脚本

支持三种数据类型：
1. daily: 日线数据（date_range模式，按日/月更新）
2. monthly: 财务数据（year_quarters模式，按季度更新）
3. single: 静态数据（single模式，不是时间序列）

用法：
  # 补充所有monthly数据（财务数据，2016-2019）
  python scripts/data/backfill_missing_data.py --mode monthly --start-year 2016 --end-year 2019

  # 补充所有daily数据（日线数据，2016-2019）
  python scripts/data/backfill_missing_data.py --mode daily --start-year 2016 --end-year 2019

  # 补充single数据（静态数据，无需年份）
  python scripts/data/backfill_missing_data.py --mode single

  # 只补充指定任务
  python scripts/data/backfill_missing_data.py --mode monthly --tasks income,balancesheet --start-year 2016 --end-year 2019

  # 测试模式
  python scripts/data/backfill_missing_data.py --mode monthly --dry-run
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from scripts.data._shared.runtime import get_tushare_root, get_tushare_token

try:
    import tushare as ts
except ImportError:
    print("错误：未安装tushare，请运行：pip install tushare")
    sys.exit(1)


class UnifiedDataBackfiller:
    """统一数据补充器

    支持三种模式：
    - daily: 日线数据（date_range）
    - monthly: 财务数据（year_quarters）
    - single: 静态数据（single）
    """

    def __init__(
        self,
        mode: str,
        start_year: int = None,
        end_year: int = None,
        tasks: list = None,
        dry_run: bool = False,
        sleep_seconds: int = 35,
    ):
        self.mode = mode
        self.start_year = start_year
        self.end_year = end_year
        self.tasks_filter = tasks
        self.dry_run = dry_run
        self.sleep_seconds = sleep_seconds
        self.data_root = get_tushare_root()
        self.config_path = ROOT / "config/tushare_tasks.yaml"

        # 加载配置
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 初始化tushare
        token = get_tushare_token()
        ts.set_token(token)
        self.pro = ts.pro_api()

        print(f"数据根目录: {self.data_root}")
        print(f"补充模式: {mode}")
        if start_year and end_year:
            print(f"补充年份: {start_year} ~ {end_year}")
        print(f"干运行模式: {dry_run}")
        print(f"请求间隔: {sleep_seconds}秒")
        if tasks:
            print(f"指定任务: {', '.join(tasks)}")
        print("=" * 80)

    def backfill_all(self):
        """补充所有缺失数据"""
        tasks = self.config.get("tasks", {})

        # 过滤任务
        if self.tasks_filter:
            tasks = {k: v for k, v in tasks.items() if k in self.tasks_filter}

        # 根据模式筛选任务
        if self.mode == "monthly":
            # 财务数据：year_quarters模式
            filtered_tasks = {k: v for k, v in tasks.items() if v.get("mode") == "year_quarters"}
        elif self.mode == "daily":
            # 日线数据：date_range模式
            filtered_tasks = {k: v for k, v in tasks.items() if v.get("mode") == "date_range"}
        elif self.mode == "single":
            # 静态数据：single模式
            filtered_tasks = {k: v for k, v in tasks.items() if v.get("mode") == "single"}
        else:
            print(f"错误：不支持的模式 '{self.mode}'")
            return

        if not filtered_tasks:
            print(f"没有找到 {self.mode} 模式的任务")
            return

        print(f"找到 {len(filtered_tasks)} 个 {self.mode} 任务:")
        for task_name in filtered_tasks.keys():
            print(f"  - {task_name}")
        print()

        start_time = time.time()

        for i, (task_name, task_config) in enumerate(filtered_tasks.items(), 1):
            print(f"\n[{i}/{len(filtered_tasks)}] 补充任务: {task_name}")
            print("-" * 80)

            try:
                if self.mode == "monthly":
                    self._backfill_monthly_task(task_name, task_config)
                elif self.mode == "daily":
                    self._backfill_daily_task(task_name, task_config)
                elif self.mode == "single":
                    self._backfill_single_task(task_name, task_config)

            except Exception as e:
                print(f"❌ 任务失败: {e}")
                import traceback

                traceback.print_exc()

                if not self.dry_run:
                    user_input = input("是否继续下一个任务？(y/n): ")
                    if user_input.lower() != "y":
                        raise

            # 任务间休息
            if i < len(filtered_tasks):
                print("\n休息10秒，避免限流...")
                if not self.dry_run:
                    time.sleep(10)

        total_elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"✅ 所有数据补充完成！总耗时: {total_elapsed/60:.1f}分钟")

    def _backfill_monthly_task(self, task_name: str, task_config: dict):
        """补充monthly模式任务（财务数据，year_quarters）

        Args:
            task_name: 任务名称
            task_config: 任务配置
        """
        api_name = task_config["api"]
        output_path = self.data_root / task_config["output"]
        quarters = task_config.get("quarters", ["0331", "0630", "0930", "1231"])

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 读取已有数据
        existing_periods = set()
        if output_path.exists():
            try:
                df_existing = pd.read_parquet(output_path)
                # 查找period字段
                period_field = None
                for field in ["end_date", "period", "f_ann_date"]:
                    if field in df_existing.columns:
                        period_field = field
                        break

                if period_field:
                    existing_periods = set(df_existing[period_field].astype(str).str[:8].unique())
                    print(f"  已有数据: {len(existing_periods)} 个季度")
            except Exception as e:
                print(f"  警告: 读取已有数据失败: {e}")

        # 生成需要补充的季度列表
        missing_periods = []
        for year in range(self.start_year, self.end_year + 1):
            for quarter in quarters:
                period = f"{year}{quarter}"
                if period not in existing_periods:
                    missing_periods.append(period)

        if not missing_periods:
            print("  ✅ 无需补充，数据已完整")
            return

        print(f"  需要补充: {len(missing_periods)} 个季度")
        print(f"  {', '.join(missing_periods[:5])}{'...' if len(missing_periods) > 5 else ''}")

        if self.dry_run:
            print(f"  [DRY RUN] 将调用API: {api_name}")
            return

        # 下载缺失的季度数据
        new_frames = []
        for period in missing_periods:
            try:
                print(f"  下载 {period}...", end=" ")

                # 调用API
                api_func = getattr(self.pro, api_name)
                df = api_func(period=period)

                if not df.empty:
                    new_frames.append(df)
                    print(f"✓ {len(df)} 条记录")
                else:
                    print("⚠️  无数据")

                time.sleep(self.sleep_seconds)

            except Exception as e:
                print(f"✗ 失败: {e}")
                raise

        # 合并并保存
        if new_frames:
            df_new = pd.concat(new_frames, ignore_index=True)

            if output_path.exists():
                # 合并已有数据
                df_existing = pd.read_parquet(output_path)
                df_all = pd.concat([df_existing, df_new], ignore_index=True)

                # 去重
                dedup_keys = task_config.get("dedup_keys", [])
                if dedup_keys:
                    df_all = df_all.drop_duplicates(subset=dedup_keys, keep="last")

                df_all.to_parquet(output_path, index=False)
                print(f"  ✅ 合并保存 {len(df_all)} 条记录（新增 {len(df_new)} 条）")
            else:
                df_new.to_parquet(output_path, index=False)
                print(f"  ✅ 保存 {len(df_new)} 条记录")

    def _backfill_daily_task(self, task_name: str, task_config: dict):
        """补充daily模式任务（日线数据，date_range）

        Args:
            task_name: 任务名称
            task_config: 任务配置
        """
        api_name = task_config["api"]
        output_path = self.data_root / task_config["output"]
        start_field = task_config.get("start_field", "start_date")
        end_field = task_config.get("end_field", "end_date")
        dedup_keys = task_config.get("dedup_keys", [])

        print(f"  API: {api_name}")
        print(f"  输出: {output_path}")
        if dedup_keys:
            print(f"  去重字段: {dedup_keys}")

        if self.dry_run:
            print(f"  [DRY RUN] 将按年份下载 {self.start_year}-{self.end_year}")
            return

        # 读取已有数据
        df_existing = None
        if output_path.exists():
            try:
                df_existing = pd.read_parquet(output_path)
                print(f"  已有数据: {len(df_existing)} 条记录")
            except Exception as e:
                print(f"  警告: 读取已有数据失败: {e}")

        # 按年份下载
        new_frames = []
        for year in range(self.start_year, self.end_year + 1):
            print(f"  下载 {year} 年数据...", end=" ")

            try:
                start_date = f"{year}0101"
                end_date = f"{year}1231"

                # 调用API
                api_func = getattr(self.pro, api_name)
                params = {start_field: start_date, end_field: end_date}

                # 添加额外参数
                if "params" in task_config:
                    params.update(task_config["params"])

                df = api_func(**params)

                if not df.empty:
                    new_frames.append(df)
                    print(f"✓ {len(df)} 条记录")
                else:
                    print("⚠️  无数据")

                time.sleep(self.sleep_seconds)

            except Exception as e:
                print(f"✗ 失败: {e}")
                raise

        # 合并并保存
        if new_frames:
            df_new = pd.concat(new_frames, ignore_index=True)

            if df_existing is not None:
                # 合并已有数据
                df_all = pd.concat([df_existing, df_new], ignore_index=True)

                # 去重
                if dedup_keys:
                    before_dedup = len(df_all)
                    df_all = df_all.drop_duplicates(subset=dedup_keys, keep="last")
                    after_dedup = len(df_all)
                    removed = before_dedup - after_dedup
                    if removed > 0:
                        print(f"  去重: 删除 {removed} 条重复记录")

                df_all.to_parquet(output_path, index=False)
                print(f"  ✅ 合并保存 {len(df_all)} 条记录（新增 {len(df_new)} 条）")
            else:
                df_new.to_parquet(output_path, index=False)
                print(f"  ✅ 保存 {len(df_new)} 条记录")
        else:
            print("  ⚠️  无新数据")

        print(f"  ✅ {task_name} 补充完成")

    def _backfill_single_task(self, task_name: str, task_config: dict):
        """补充single模式任务（静态数据，single）

        Args:
            task_name: 任务名称
            task_config: 任务配置
        """
        api_name = task_config["api"]
        output_path = self.data_root / task_config["output"]
        dedup_keys = task_config.get("dedup_keys", [])

        print(f"  API: {api_name}")
        print(f"  输出: {output_path}")
        if dedup_keys:
            print(f"  去重字段: {dedup_keys}")

        if output_path.exists():
            print("  ✅ 文件已存在，跳过")
            return

        if self.dry_run:
            print(f"  [DRY RUN] 将调用API: {api_name}")
            return

        try:
            print("  下载数据...", end=" ")

            # 调用API
            api_func = getattr(self.pro, api_name)
            params = task_config.get("params", {})
            df = api_func(**params)

            if not df.empty:
                # 去重（如果配置了dedup_keys）
                if dedup_keys:
                    before_dedup = len(df)
                    df = df.drop_duplicates(subset=dedup_keys, keep="last")
                    after_dedup = len(df)
                    removed = before_dedup - after_dedup
                    if removed > 0:
                        print(f"去重: 删除 {removed} 条重复记录，", end="")

                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(output_path, index=False)
                print(f"✓ {len(df)} 条记录")
            else:
                print("⚠️  无数据")

        except Exception as e:
            print(f"✗ 失败: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="统一数据补充脚本")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["daily", "monthly", "single"],
        help="补充模式: daily(日线), monthly(财务), single(静态)",
    )
    parser.add_argument("--start-year", type=int, help="起始年份（daily/monthly模式必需）")
    parser.add_argument("--end-year", type=int, help="结束年份（daily/monthly模式必需）")
    parser.add_argument("--tasks", type=str, help="指定任务（逗号分隔），如: income,balancesheet,cashflow")
    parser.add_argument("--dry-run", action="store_true", help="测试模式（不实际下载）")
    parser.add_argument("--sleep-seconds", type=int, default=35, help="请求间隔（秒），默认35秒")

    args = parser.parse_args()

    # 验证参数
    if args.mode in ["daily", "monthly"]:
        if not args.start_year or not args.end_year:
            parser.error(f"{args.mode} 模式需要指定 --start-year 和 --end-year")

    tasks = args.tasks.split(",") if args.tasks else None

    backfiller = UnifiedDataBackfiller(
        mode=args.mode,
        start_year=args.start_year,
        end_year=args.end_year,
        tasks=tasks,
        dry_run=args.dry_run,
        sleep_seconds=args.sleep_seconds,
    )

    backfiller.backfill_all()


if __name__ == "__main__":
    main()
