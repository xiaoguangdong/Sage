#!/usr/bin/env python3
"""
优化版Tushare批量下载工具（v2）

优化策略：
1. margin接口：一次性传全部日期范围，使用offset/limit分页获取
2. daily_basic接口：一次性传全部日期范围，使用offset/limit分页获取
3. API调用间隔：60秒
"""

import sys
from pathlib import Path
import json
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_tushare_token, setup_logger
from sage_app.data.data_provider import DataProvider

logger = setup_logger(Path(__file__).stem)


class OptimizedBatchDownloader:
    """优化版批量下载器 - 使用offset/limit分页"""

    def __init__(self, tushare_token: str, start_date: str = "2020-01-01", end_date: str = "2024-12-31"):
        """
        初始化批量下载器

        Args:
            tushare_token: Tushare token
            start_date: 开始日期
            end_date: 结束日期
        """
        self.tushare_token = tushare_token
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = "data/tushare"

        # API调用间隔（秒）
        self.api_delay = 60  # 60秒

        # 状态文件
        self.state_file = os.path.join(self.output_dir, "download_state_optimized.json")

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def load_stock_list(self, csv_path: str = "data/tushare/filtered_stocks_list.csv") -> pd.DataFrame:
        """加载股票列表（已过滤中小板和ST股票）"""
        logger.info(f"加载股票列表: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"找到 {len(df)} 只股票（已排除中小板和ST股票）")
        return df

    def save_state(self, state: Dict):
        """保存下载状态"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self) -> Dict:
        """加载下载状态"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            logger.info(f"从断点恢复")
            return state
        return {
            'margin_offset': 0,
            'daily_basic_offset': 0,
            'results': []
        }

    def download_margin_all(self, provider: DataProvider, resume: bool = False):
        """
        批量下载所有margin数据（使用offset/limit分页）

        Args:
            provider: DataProvider实例
            resume: 是否断点续传
        """
        logger.info("=" * 70)
        logger.info("开始批量下载 margin 数据（全市场，分页）")
        logger.info("=" * 70)

        state = self.load_state() if resume else {}
        offset = state.get('margin_offset', 0)
        limit = 4000  # margin接口单次最大返回4000条

        logger.info(f"起始offset: {offset}")
        logger.info(f"每次获取: {limit} 条")

        all_data = []
        batch_count = 0

        while True:
            logger.info(f"\n批次 {batch_count + 1}: offset={offset}, limit={limit}")
            logger.info(f"等待 {self.api_delay} 秒...")
            time.sleep(self.api_delay)

            try:
                df = provider.ts_pro.margin(
                    start_date=self.start_date.replace('-', ''),
                    end_date=self.end_date.replace('-', ''),
                    limit=limit,
                    offset=offset
                )

                if df.empty:
                    logger.info(f"  数据为空，下载完成")
                    break

                logger.info(f"  ✓ 获取 {len(df)} 条记录")
                all_data.append(df)
                
                # 更新offset
                offset += len(df)
                batch_count += 1

                # 保存状态
                state['margin_offset'] = offset
                self.save_state(state)

                # 如果获取的数据少于limit，说明已经是最后一页了
                if len(df) < limit:
                    logger.info(f"  数据量少于limit，下载完成")
                    break

            except Exception as e:
                logger.error(f"  ✗ 下载失败: {e}")
                # 失败时也保存当前offset，下次继续
                state['margin_offset'] = offset
                self.save_state(state)
                break

        # 合并所有批次的数据
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('trade_date')

            # 保存合并后的文件
            output_path = os.path.join(self.output_dir, "margin_market_all.parquet")
            combined_df.to_parquet(output_path)

            logger.info("\n" + "=" * 70)
            logger.info("margin 数据下载完成！")
            logger.info("=" * 70)
            logger.info(f"  总记录数: {len(combined_df)}")
            logger.info(f"  日期范围: {combined_df['trade_date'].min()} - {combined_df['trade_date'].max()}")
            logger.info(f"  交易所数量: {combined_df['exchange_id'].nunique()}")
            logger.info(f"  保存到: {output_path}")

    def download_daily_basic_all(self, provider: DataProvider, resume: bool = False):
        """
        批量下载所有daily_basic数据（使用offset/limit分页）

        Args:
            provider: DataProvider实例
            resume: 是否断点续传
        """
        logger.info("=" * 70)
        logger.info("开始批量下载 daily_basic 数据（全市场，分页）")
        logger.info("=" * 70)

        state = self.load_state() if resume else {}
        offset = state.get('daily_basic_offset', 0)
        limit = 6000  # daily_basic接口单次最大返回6000条

        logger.info(f"起始offset: {offset}")
        logger.info(f"每次获取: {limit} 条")

        all_data = []
        batch_count = 0

        while True:
            logger.info(f"\n批次 {batch_count + 1}: offset={offset}, limit={limit}")
            logger.info(f"等待 {self.api_delay} 秒...")
            time.sleep(self.api_delay)

            try:
                df = provider.ts_pro.daily_basic(
                    start_date=self.start_date.replace('-', ''),
                    end_date=self.end_date.replace('-', ''),
                    limit=limit,
                    offset=offset
                )

                if df.empty:
                    logger.info(f"  数据为空，下载完成")
                    break

                logger.info(f"  ✓ 获取 {len(df)} 条记录")
                all_data.append(df)
                
                # 更新offset
                offset += len(df)
                batch_count += 1

                # 保存状态
                state['daily_basic_offset'] = offset
                self.save_state(state)

                # 如果获取的数据少于limit，说明已经是最后一页了
                if len(df) < limit:
                    logger.info(f"  数据量少于limit，下载完成")
                    break

            except Exception as e:
                logger.error(f"  ✗ 下载失败: {e}")
                # 失败时也保存当前offset，下次继续
                state['daily_basic_offset'] = offset
                self.save_state(state)
                break

        # 合并所有批次的数据
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['ts_code', 'trade_date'])

            # 保存合并后的文件
            output_path = os.path.join(self.output_dir, "daily_basic_all.parquet")
            combined_df.to_parquet(output_path)

            logger.info("\n" + "=" * 70)
            logger.info("daily_basic 数据下载完成！")
            logger.info("=" * 70)
            logger.info(f"  总记录数: {len(combined_df)}")
            logger.info(f"  股票数量: {combined_df['ts_code'].nunique()}")
            logger.info(f"  日期范围: {combined_df['trade_date'].min()} - {combined_df['trade_date'].max()}")
            logger.info(f"  保存到: {output_path}")

    def download_all(self, stocks_df: pd.DataFrame, resume: bool = True):
        """
        下载所有数据

        Args:
            stocks_df: 股票列表
            resume: 是否断点续传
        """
        provider = DataProvider(tushare_token=self.tushare_token)

        # 1. 批量下载margin数据（全市场，分页）
        self.download_margin_all(provider, resume=resume)

        # 2. 批量下载daily_basic数据（全市场，分页）
        self.download_daily_basic_all(provider, resume=resume)

        logger.info("\n" + "=" * 70)
        logger.info("批量数据下载完成！")
        logger.info("=" * 70)
        logger.info("提示: 财务指标数据需要单独使用 download_fina_indicator.py 下载")
        logger.info("=" * 70)


def main():
    """主函数"""
    # 配置
    TUSHARE_TOKEN = get_tushare_token()
    START_DATE = "2020-01-01"
    END_DATE = "2026-02-09"

    logger.info("=" * 70)
    logger.info("优化版Tushare批量下载工具（v2）")
    logger.info("=" * 70)
    logger.info("优化策略:")
    logger.info("  1. margin: 一次性传全部日期范围，使用offset/limit分页获取")
    logger.info("  2. daily_basic: 一次性传全部日期范围，使用offset/limit分页获取")
    logger.info("  3. API调用间隔: 60秒")
    logger.info("  4. 支持断点续传")
    logger.info("  5. 注意: 财务指标数据需要单独使用 download_fina_indicator.py 下载")
    logger.info("=" * 70)

    # 创建下载器
    downloader = OptimizedBatchDownloader(
        tushare_token=TUSHARE_TOKEN,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # 加载股票列表（用于统计）
    stocks_df = downloader.load_stock_list()

    # 开始下载
    downloader.download_all(
        stocks_df=stocks_df,
        resume=True
    )


if __name__ == "__main__":
    main()
