#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
历史数据补充脚本（2016-2020）

用于补充长周期回测所需的历史数据
参考tushare_downloader.py的限流策略

用法：
  # 测试模式（不实际下载）
  python scripts/data/backfill_historical_data.py --start-year 2016 --end-year 2016 --dry-run

  # 正式下载（默认60秒间隔）
  python scripts/data/backfill_historical_data.py --start-year 2016 --end-year 2020

  # 自定义间隔（积分不足时可增加到120秒）
  python scripts/data/backfill_historical_data.py --start-year 2016 --end-year 2020 --sleep-seconds 120
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data._shared.runtime import get_tushare_root, get_tushare_token

try:
    import tushare as ts
except ImportError:
    print("错误：未安装tushare，请运行：pip install tushare")
    sys.exit(1)


class HistoricalDataBackfiller:
    """历史数据补充器

    参考tushare_downloader.py的设计：
    - 默认sleep_seconds=60（比downloader的40秒更保守）
    - 支持按年份/季度分批下载
    - 自动跳过已存在文件
    - 支持dry-run测试
    """

    def __init__(
        self,
        start_year: int,
        end_year: int,
        dry_run: bool = False,
        sleep_seconds: int = 60,
    ):
        self.start_year = start_year
        self.end_year = end_year
        self.dry_run = dry_run
        self.sleep_seconds = sleep_seconds
        self.data_root = get_tushare_root()

        # 配置日志
        self._setup_logging()

        # 初始化tushare
        token = get_tushare_token()
        ts.set_token(token)
        self.pro = ts.pro_api()

        self.log(f"数据根目录: {self.data_root}")
        self.log(f"补充年份: {start_year} ~ {end_year}")
        self.log(f"干运行模式: {dry_run}")
        self.log(f"请求间隔: {sleep_seconds}秒")
        self.log("=" * 60)

    def _setup_logging(self):
        """配置日志（带时间戳）"""
        log_dir = ROOT / "logs" / "data"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"backfill_{timestamp}.log"

        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        )

        self.logger = logging.getLogger(__name__)
        self.log(f"日志文件: {log_file}")

    def log(self, message: str, level: str = "INFO"):
        """统一日志输出"""
        if level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "SUCCESS":
            self.logger.info(f"✓ {message}")

    def backfill_all(self):
        """按顺序补充所有数据

        数据类型优先级：
        1. 日线数据（必需）
        2. 每日基本面（包含股息率）
        3. 财务三表（利润表、资产负债表、现金流量表）
        4. 财务指标（ROE、ROIC等）
        5. 分红数据（连续分红年数）
        6. 沪深300成分股（股票池）
        """
        tasks = [
            ("日线数据 (daily)", self.backfill_daily),
            ("每日基本面 (daily_basic)", self.backfill_daily_basic),
            ("财务指标 (fina_indicator)", self.backfill_fina_indicator),
            ("利润表 (income)", self.backfill_income),
            ("资产负债表 (balancesheet)", self.backfill_balancesheet),
            ("现金流量表 (cashflow)", self.backfill_cashflow),
            ("分红数据 (dividend)", self.backfill_dividend),
            ("沪深300成分股 (hs300_constituents)", self.backfill_hs300_constituents),
        ]

        start_time = time.time()
        total_tasks = len(tasks)

        for i, (name, func) in enumerate(tasks, 1):
            self.log(f"\n[{i}/{total_tasks}] 开始补充: {name}")
            self.log("-" * 60)

            task_start = time.time()
            try:
                func()
                task_elapsed = time.time() - task_start
                self.log(f"{name} 补充完成，耗时 {task_elapsed:.1f}秒", "SUCCESS")
            except Exception as e:
                self.log(f"{name} 补充失败: {e}", "ERROR")
                import traceback

                self.logger.error(traceback.format_exc())

                if not self.dry_run:
                    user_input = input("是否继续下一个任务？(y/n): ")
                    if user_input.lower() != "y":
                        raise

            # 任务间休息，避免限流
            if i < total_tasks:
                wait_time = 10
                self.log(f"休息{wait_time}秒，避免限流...")
                if not self.dry_run:
                    time.sleep(wait_time)

        total_elapsed = time.time() - start_time
        self.log("=" * 60)
        self.log(f"所有数据补充完成！总耗时: {total_elapsed/60:.1f}分钟", "SUCCESS")

    def backfill_daily(self):
        """补充日线数据

        API: pro.daily()
        限流: 每次请求后休息sleep_seconds
        分批: 按年份
        """
        daily_dir = self.data_root / "daily"
        daily_dir.mkdir(exist_ok=True)

        for year in range(self.start_year, self.end_year + 1):
            file_path = daily_dir / f"daily_{year}.parquet"

            if file_path.exists():
                self.log(f"  跳过 {year}: 文件已存在")
                continue

            self.log(f"  下载 {year} 年日线数据...")

            if self.dry_run:
                self.log(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            start_date = f"{year}0101"
            end_date = f"{year}1231"

            try:
                df = self.pro.daily(
                    start_date=start_date,
                    end_date=end_date,
                )

                if df.empty:
                    self.log(f"    警告: {year} 年无数据")
                else:
                    df.to_parquet(file_path, index=False)
                    self.log(f"    ✓ 保存 {len(df)} 条记录")

            except Exception as e:
                self.log(f"    ✗ 下载失败: {e}")
                raise

            time.sleep(self.sleep_seconds)

    def backfill_daily_basic(self):
        """补充每日基本面数据（包含股息率dv_ratio, dv_ttm）

        API: pro.daily_basic()
        限流: 每次请求后休息sleep_seconds
        分批: 按年份+月份（数据量大）
        """
        daily_basic_dir = self.data_root / "daily_basic"
        daily_basic_dir.mkdir(exist_ok=True)

        for year in range(self.start_year, self.end_year + 1):
            file_path = daily_basic_dir / f"daily_basic_{year}.parquet"

            if file_path.exists():
                self.log(f"  跳过 {year}: 文件已存在")
                continue

            self.log(f"  下载 {year} 年每日基本面数据...")

            if self.dry_run:
                self.log(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            # 按月下载（避免单次数据量过大）
            frames = []
            for month in range(1, 13):
                start_date = f"{year}{month:02d}01"
                # 月末日期简化处理
                if month == 12:
                    end_date = f"{year}1231"
                else:
                    end_date = f"{year}{month+1:02d}01"

                try:
                    df = self.pro.daily_basic(
                        start_date=start_date,
                        end_date=end_date,
                    )

                    if not df.empty:
                        frames.append(df)
                        self.log(f"    ✓ {year}-{month:02d}: {len(df)} 条记录")

                    time.sleep(self.sleep_seconds)

                except Exception as e:
                    self.log(f"    ✗ {year}-{month:02d} 下载失败: {e}")
                    raise

            if frames:
                df_all = pd.concat(frames, ignore_index=True)
                df_all.to_parquet(file_path, index=False)
                self.log(f"    ✓ 合并保存 {len(df_all)} 条记录")

    def backfill_fina_indicator(self):
        """补充财务指标数据（ROE、ROIC、毛利率等）

        API: pro.fina_indicator()
        限流: 每次请求后休息sleep_seconds
        分批: 按年份
        """
        fina_dir = self.data_root / "fundamental"
        fina_dir.mkdir(exist_ok=True)

        for year in range(self.start_year, self.end_year + 1):
            file_path = fina_dir / f"fina_indicator_{year}.parquet"

            if file_path.exists():
                self.log(f"  跳过 {year}: 文件已存在")
                continue

            self.log(f"  下载 {year} 年财务指标数据...")

            if self.dry_run:
                self.log(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            # 按季度下载
            frames = []
            for period in [f"{year}0331", f"{year}0630", f"{year}0930", f"{year}1231"]:
                try:
                    df = self.pro.fina_indicator(period=period)

                    if not df.empty:
                        frames.append(df)
                        self.log(f"    ✓ {period}: {len(df)} 条记录")

                    time.sleep(self.sleep_seconds)

                except Exception as e:
                    self.log(f"    ✗ {period} 下载失败: {e}")
                    raise

            if frames:
                df_all = pd.concat(frames, ignore_index=True)
                df_all.to_parquet(file_path, index=False)
                self.log(f"    ✓ 合并保存 {len(df_all)} 条记录")

    def backfill_income(self):
        """补充利润表数据

        API: pro.income_vip()
        限流: 每次请求后休息sleep_seconds
        分批: 按年份+季度
        """
        income_dir = self.data_root / "fundamental" / "income"
        income_dir.mkdir(parents=True, exist_ok=True)

        for year in range(self.start_year, self.end_year + 1):
            file_path = income_dir / f"income_{year}.parquet"

            if file_path.exists():
                self.log(f"  跳过 {year}: 文件已存在")
                continue

            self.log(f"  下载 {year} 年利润表数据...")

            if self.dry_run:
                self.log(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            frames = []
            for period in [f"{year}0331", f"{year}0630", f"{year}0930", f"{year}1231"]:
                try:
                    df = self.pro.income_vip(period=period)

                    if not df.empty:
                        frames.append(df)
                        self.log(f"    ✓ {period}: {len(df)} 条记录")

                    time.sleep(self.sleep_seconds)

                except Exception as e:
                    self.log(f"    ✗ {period} 下载失败: {e}")
                    raise

            if frames:
                df_all = pd.concat(frames, ignore_index=True)
                df_all.to_parquet(file_path, index=False)
                self.log(f"    ✓ 合并保存 {len(df_all)} 条记录")

    def backfill_balancesheet(self):
        """补充资产负债表数据

        API: pro.balancesheet_vip()
        限流: 每次请求后休息sleep_seconds
        分批: 按年份+季度
        """
        balance_dir = self.data_root / "fundamental" / "balancesheet"
        balance_dir.mkdir(parents=True, exist_ok=True)

        for year in range(self.start_year, self.end_year + 1):
            file_path = balance_dir / f"balancesheet_{year}.parquet"

            if file_path.exists():
                self.log(f"  跳过 {year}: 文件已存在")
                continue

            self.log(f"  下载 {year} 年资产负债表数据...")

            if self.dry_run:
                self.log(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            frames = []
            for period in [f"{year}0331", f"{year}0630", f"{year}0930", f"{year}1231"]:
                try:
                    df = self.pro.balancesheet_vip(period=period)

                    if not df.empty:
                        frames.append(df)
                        self.log(f"    ✓ {period}: {len(df)} 条记录")

                    time.sleep(self.sleep_seconds)

                except Exception as e:
                    self.log(f"    ✗ {period} 下载失败: {e}")
                    raise

            if frames:
                df_all = pd.concat(frames, ignore_index=True)
                df_all.to_parquet(file_path, index=False)
                self.log(f"    ✓ 合并保存 {len(df_all)} 条记录")

    def backfill_cashflow(self):
        """补充现金流量表数据

        API: pro.cashflow_vip()
        限流: 每次请求后休息sleep_seconds
        分批: 按年份+季度
        """
        cashflow_dir = self.data_root / "fundamental" / "cashflow"
        cashflow_dir.mkdir(parents=True, exist_ok=True)

        for year in range(self.start_year, self.end_year + 1):
            file_path = cashflow_dir / f"cashflow_{year}.parquet"

            if file_path.exists():
                self.log(f"  跳过 {year}: 文件已存在")
                continue

            self.log(f"  下载 {year} 年现金流量表数据...")

            if self.dry_run:
                self.log(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            frames = []
            for period in [f"{year}0331", f"{year}0630", f"{year}0930", f"{year}1231"]:
                try:
                    df = self.pro.cashflow_vip(period=period)

                    if not df.empty:
                        frames.append(df)
                        self.log(f"    ✓ {period}: {len(df)} 条记录")

                    time.sleep(self.sleep_seconds)

                except Exception as e:
                    self.log(f"    ✗ {period} 下载失败: {e}")
                    raise

            if frames:
                df_all = pd.concat(frames, ignore_index=True)
                df_all.to_parquet(file_path, index=False)
                self.log(f"    ✓ 合并保存 {len(df_all)} 条记录")

    def backfill_dividend(self):
        """补充分红数据（用于计算连续分红年数）

        API: pro.dividend()
        限流: 每次请求后休息sleep_seconds
        分批: 按年份
        """
        dividend_dir = self.data_root / "fundamental" / "dividend"
        dividend_dir.mkdir(parents=True, exist_ok=True)

        for year in range(self.start_year, self.end_year + 1):
            file_path = dividend_dir / f"dividend_{year}.parquet"

            if file_path.exists():
                self.log(f"  跳过 {year}: 文件已存在")
                continue

            self.log(f"  下载 {year} 年分红数据...")

            if self.dry_run:
                self.log(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            try:
                # 按实施日期查询
                df = self.pro.dividend(
                    imp_ann_date=f"{year}0101",
                    end_date=f"{year}1231",
                )

                if not df.empty:
                    df.to_parquet(file_path, index=False)
                    self.log(f"    ✓ 保存 {len(df)} 条记录")
                else:
                    self.log(f"    警告: {year} 年无分红数据")

            except Exception as e:
                self.log(f"    ✗ 下载失败: {e}")
                raise

            time.sleep(self.sleep_seconds)

    def backfill_hs300_constituents(self):
        """补充沪深300成分股数据

        API: pro.index_weight()
        限流: 每次请求后休息sleep_seconds
        分批: 按年份
        """
        constituents_dir = self.data_root / "constituents"
        constituents_dir.mkdir(exist_ok=True)

        for year in range(self.start_year, self.end_year + 1):
            file_path = constituents_dir / f"hs300_constituents_{year}.parquet"

            if file_path.exists():
                self.log(f"  跳过 {year}: 文件已存在")
                continue

            self.log(f"  下载 {year} 年沪深300成分股数据...")

            if self.dry_run:
                self.log(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            start_date = f"{year}0101"
            end_date = f"{year}1231"

            try:
                df = self.pro.index_weight(
                    index_code="000300.SH",
                    start_date=start_date,
                    end_date=end_date,
                )

                if not df.empty:
                    df.to_parquet(file_path, index=False)
                    self.log(f"    ✓ 保存 {len(df)} 条记录")
                else:
                    self.log(f"    警告: {year} 年无成分股数据")

            except Exception as e:
                self.log(f"    ✗ 下载失败: {e}")
                raise

            time.sleep(self.sleep_seconds)


def main():
    parser = argparse.ArgumentParser(description="补充历史数据（2016-2020）")
    parser.add_argument("--start-year", type=int, default=2016, help="起始年份")
    parser.add_argument("--end-year", type=int, default=2020, help="结束年份")
    parser.add_argument("--dry-run", action="store_true", help="测试模式（不实际下载）")
    parser.add_argument("--sleep-seconds", type=int, default=60, help="请求间隔（秒），默认60秒")

    args = parser.parse_args()

    backfiller = HistoricalDataBackfiller(
        start_year=args.start_year,
        end_year=args.end_year,
        dry_run=args.dry_run,
        sleep_seconds=args.sleep_seconds,
    )

    backfiller.backfill_all()


if __name__ == "__main__":
    main()
