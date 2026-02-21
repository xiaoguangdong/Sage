#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
下载财务三表数据（利润表、资产负债表、现金流量表）

用法：
  python scripts/data/download_financial_statements.py --start-year 2020 --end-year 2024
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import tushare as ts

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data._shared.runtime import get_tushare_root, get_tushare_token, log_task_summary, setup_logger


class FinancialStatementsDownloader:
    """财务三表下载器"""

    def __init__(
        self,
        start_year: int,
        end_year: int,
        sleep_seconds: int = 60,
        dry_run: bool = False,
    ):
        self.start_year = start_year
        self.end_year = end_year
        self.sleep_seconds = sleep_seconds
        self.dry_run = dry_run

        # 初始化Tushare
        ts.set_token(get_tushare_token())
        self.pro = ts.pro_api()

        # 数据目录
        self.data_root = get_tushare_root()
        self.income_dir = self.data_root / "fundamental" / "income"
        self.balance_dir = self.data_root / "fundamental" / "balancesheet"
        self.cashflow_dir = self.data_root / "fundamental" / "cashflow"
        self.dividend_dir = self.data_root / "fundamental" / "dividend"

        # 创建目录
        self.income_dir.mkdir(parents=True, exist_ok=True)
        self.balance_dir.mkdir(parents=True, exist_ok=True)
        self.cashflow_dir.mkdir(parents=True, exist_ok=True)
        self.dividend_dir.mkdir(parents=True, exist_ok=True)

        # 配置日志
        self.logger = setup_logger("download_financial_statements", module="data")

    def download_income(self):
        """下载利润表"""
        self.logger.info("\n[1/4] 下载利润表 (income)")
        self.logger.info("-" * 60)

        for year in range(self.start_year, self.end_year + 1):
            file_path = self.income_dir / f"income_{year}.parquet"

            if file_path.exists():
                self.logger.info(f"  跳过 {year}: 文件已存在")
                continue

            self.logger.info(f"  下载 {year} 年利润表...")

            if self.dry_run:
                self.logger.info(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            frames = []
            for period in [f"{year}0331", f"{year}0630", f"{year}0930", f"{year}1231"]:
                try:
                    df = self.pro.income(period=period)

                    if not df.empty:
                        frames.append(df)
                        self.logger.info(f"    ✓ {period}: {len(df)} 条记录")

                    time.sleep(self.sleep_seconds)

                except Exception as e:
                    self.logger.error(f"    ✗ {period} 下载失败: {e}")
                    raise

            if frames:
                df_all = pd.concat(frames, ignore_index=True)
                df_all.to_parquet(file_path, index=False)
                self.logger.info(f"    ✓ 合并保存 {len(df_all)} 条记录")

    def download_balancesheet(self):
        """下载资产负债表"""
        self.logger.info("\n[2/4] 下载资产负债表 (balancesheet)")
        self.logger.info("-" * 60)

        for year in range(self.start_year, self.end_year + 1):
            file_path = self.balance_dir / f"balancesheet_{year}.parquet"

            if file_path.exists():
                self.logger.info(f"  跳过 {year}: 文件已存在")
                continue

            self.logger.info(f"  下载 {year} 年资产负债表...")

            if self.dry_run:
                self.logger.info(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            frames = []
            for period in [f"{year}0331", f"{year}0630", f"{year}0930", f"{year}1231"]:
                try:
                    df = self.pro.balancesheet(period=period)

                    if not df.empty:
                        frames.append(df)
                        self.logger.info(f"    ✓ {period}: {len(df)} 条记录")

                    time.sleep(self.sleep_seconds)

                except Exception as e:
                    self.logger.error(f"    ✗ {period} 下载失败: {e}")
                    raise

            if frames:
                df_all = pd.concat(frames, ignore_index=True)
                df_all.to_parquet(file_path, index=False)
                self.logger.info(f"    ✓ 合并保存 {len(df_all)} 条记录")

    def download_cashflow(self):
        """下载现金流量表"""
        self.logger.info("\n[3/4] 下载现金流量表 (cashflow)")
        self.logger.info("-" * 60)

        for year in range(self.start_year, self.end_year + 1):
            file_path = self.cashflow_dir / f"cashflow_{year}.parquet"

            if file_path.exists():
                self.logger.info(f"  跳过 {year}: 文件已存在")
                continue

            self.logger.info(f"  下载 {year} 年现金流量表...")

            if self.dry_run:
                self.logger.info(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            frames = []
            for period in [f"{year}0331", f"{year}0630", f"{year}0930", f"{year}1231"]:
                try:
                    df = self.pro.cashflow(period=period)

                    if not df.empty:
                        frames.append(df)
                        self.logger.info(f"    ✓ {period}: {len(df)} 条记录")

                    time.sleep(self.sleep_seconds)

                except Exception as e:
                    self.logger.error(f"    ✗ {period} 下载失败: {e}")
                    raise

            if frames:
                df_all = pd.concat(frames, ignore_index=True)
                df_all.to_parquet(file_path, index=False)
                self.logger.info(f"    ✓ 合并保存 {len(df_all)} 条记录")

    def download_dividend(self):
        """下载分红数据"""
        self.logger.info("\n[4/4] 下载分红数据 (dividend)")
        self.logger.info("-" * 60)

        for year in range(self.start_year, self.end_year + 1):
            file_path = self.dividend_dir / f"dividend_{year}.parquet"

            if file_path.exists():
                self.logger.info(f"  跳过 {year}: 文件已存在")
                continue

            self.logger.info(f"  下载 {year} 年分红数据...")

            if self.dry_run:
                self.logger.info(f"    [DRY RUN] 将保存到: {file_path}")
                continue

            try:
                df = self.pro.dividend(
                    imp_ann_date=f"{year}0101",
                    end_date=f"{year}1231",
                )

                if not df.empty:
                    df.to_parquet(file_path, index=False)
                    self.logger.info(f"    ✓ 保存 {len(df)} 条记录")
                else:
                    self.logger.warning(f"    警告: {year} 年无分红数据")

            except Exception as e:
                self.logger.error(f"    ✗ 下载失败: {e}")
                raise

            time.sleep(self.sleep_seconds)

    def run(self):
        """执行下载"""
        self.logger.info("=" * 60)
        self.logger.info("财务三表数据下载")
        self.logger.info("=" * 60)
        self.logger.info(f"年份范围: {self.start_year} ~ {self.end_year}")
        self.logger.info(f"请求间隔: {self.sleep_seconds}秒")
        self.logger.info(f"干运行模式: {self.dry_run}")
        self.logger.info("=" * 60)

        try:
            self.download_income()
            self.download_balancesheet()
            self.download_cashflow()
            self.download_dividend()

            self.logger.info("\n" + "=" * 60)
            self.logger.info("✓ 所有数据下载完成！")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"\n✗ 下载失败: {e}")
            raise


def main():
    start_time = datetime.now().timestamp()
    failure_reason = None
    parser = argparse.ArgumentParser(description="下载财务三表数据")
    parser.add_argument("--start-year", type=int, default=2020, help="起始年份")
    parser.add_argument("--end-year", type=int, default=2024, help="结束年份")
    parser.add_argument("--sleep-seconds", type=int, default=60, help="请求间隔（秒）")
    parser.add_argument("--dry-run", action="store_true", help="干运行模式（不实际下载）")

    try:
        args = parser.parse_args()

        downloader = FinancialStatementsDownloader(
            start_year=args.start_year,
            end_year=args.end_year,
            sleep_seconds=args.sleep_seconds,
            dry_run=args.dry_run,
        )

        downloader.run()
    except Exception as exc:
        failure_reason = str(exc)
        raise
    finally:
        log_task_summary(
            setup_logger("download_financial_statements", module="data"),
            task_name="download_financial_statements",
            window=f"{args.start_year}~{args.end_year}" if "args" in locals() else None,
            elapsed_s=datetime.now().timestamp() - start_time,
            error=failure_reason,
        )


if __name__ == "__main__":
    main()
