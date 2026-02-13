#!/usr/bin/env python3
"""
财务指标数据下载脚本

单独下载fina_indicator数据，因为需要逐个股票获取，与其他批量接口逻辑不同
"""

import os
import sys
import json
import time
import logging
import pandas as pd
from typing import Dict

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ml_stock_forecast.data.data_provider import DataProvider

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/tushare/download_fina_indicator.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class FinaIndicatorDownloader:
    """财务指标下载器"""

    def __init__(self, tushare_token: str, start_date: str = "2020-01-01", end_date: str = "2024-12-31"):
        """
        初始化下载器

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
        self.state_file = os.path.join(self.output_dir, "fina_indicator_state.json")

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
            logger.info(f"从断点恢复: 已完成 {state['completed_count']} 只股票")
            return state
        return {
            'completed_count': 0,
            'current_stock_idx': 0,
            'results': []
        }

    def download_single_stock(self, provider: DataProvider, baostock_code: str, tushare_code: str, stock_name: str) -> Dict:
        """
        下载单个股票的fina_indicator数据

        Args:
            provider: DataProvider实例
            baostock_code: Baostock代码
            tushare_code: Tushare代码
            stock_name: 股票名称

        Returns:
            下载结果
        """
        results = {
            'baostock_code': baostock_code,
            'tushare_code': tushare_code,
            'stock_name': stock_name,
            'fina_indicator': None,
            'success': False,
            'error': None
        }

        # 下载财务指标数据
        logger.info(f"  准备下载 fina_indicator 数据，等待 {self.api_delay} 秒...")
        time.sleep(self.api_delay)
        try:
            df = provider.get_fina_indicator_tushare(tushare_code, self.start_date, self.end_date)
            if not df.empty:
                output_path = os.path.join(
                    self.output_dir,
                    f"fina_indicator_{baostock_code.replace('.', '_')}_{self.start_date.replace('-', '')}_{self.end_date.replace('-', '')}.parquet"
                )
                df.to_parquet(output_path)
                results['fina_indicator'] = len(df)
                results['success'] = True
                logger.info(f"  ✓ fina_indicator: {len(df)} 条记录")
            else:
                logger.info(f"  ✗ fina_indicator: 数据为空")
        except Exception as e:
            logger.error(f"  ✗ fina_indicator: {e}")
            results['error'] = str(e)

        return results

    def download_all(self, stocks_df: pd.DataFrame, resume: bool = True):
        """
        下载所有股票的fina_indicator数据

        Args:
            stocks_df: 股票列表
            resume: 是否断点续传
        """
        logger.info("=" * 70)
        logger.info("开始下载财务指标数据（fina_indicator）")
        logger.info("=" * 70)

        provider = DataProvider(tushare_token=self.tushare_token)

        state = self.load_state() if resume else {}
        start_idx = state.get('current_stock_idx', 0)

        total_stocks = len(stocks_df)
        logger.info(f"总股票数: {total_stocks}")
        logger.info(f"从第 {start_idx + 1} 只股票开始下载")
        logger.info(f"预计时间: {total_stocks * self.api_delay / 3600:.1f} 小时")

        for i in range(start_idx, total_stocks):
            row = stocks_df.iloc[i]
            baostock_code = row['code']
            tushare_code = row['tushare_code']
            stock_name = row.get('name', 'Unknown')

            logger.info(f"\n[{i+1}/{total_stocks}] {stock_name} ({baostock_code})")

            result = self.download_single_stock(provider, baostock_code, tushare_code, stock_name)
            state['results'].append(result)

            # 每10只股票保存一次状态
            if (i + 1) % 10 == 0:
                state['current_stock_idx'] = i + 1
                state['completed_count'] = i + 1
                self.save_state(state)

                # 显示进度
                success_count = sum(1 for r in state['results'] if r['success'])
                logger.info(f"\n进度: {i+1}/{total_stocks} ({(i+1)/total_stocks*100:.1f}%)")
                logger.info(f"成功: {success_count}/{i+1}")

        # 保存最终状态
        state['current_stock_idx'] = total_stocks
        state['completed_count'] = total_stocks
        self.save_state(state)

        # 统计结果
        logger.info("\n" + "=" * 70)
        logger.info("下载完成！")
        logger.info("=" * 70)

        success_count = sum(1 for r in state['results'] if r['success'])
        logger.info(f"成功: {success_count}/{total_stocks} 只股票")

        # 保存详细结果
        results_df = pd.DataFrame(state['results'])
        results_df.to_csv(os.path.join(self.output_dir, "fina_indicator_results.csv"), index=False, encoding='utf-8-sig')
        logger.info(f"详细结果已保存: {os.path.join(self.output_dir, 'fina_indicator_results.csv')}")


def main():
    """主函数"""
    # 配置
    TUSHARE_TOKEN = "2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e"
    START_DATE = "20200101"
    END_DATE = "20260209"

    logger.info("=" * 70)
    logger.info("财务指标数据下载工具")
    logger.info("=" * 70)
    logger.info("配置:")
    logger.info(f"  日期范围: {START_DATE} - {END_DATE}")
    logger.info(f"  API调用间隔: 60秒")
    logger.info("=" * 70)

    # 创建下载器
    downloader = FinaIndicatorDownloader(
        tushare_token=TUSHARE_TOKEN,
        start_date=START_DATE,
        end_date=END_DATE
    )

    # 加载股票列表
    stocks_df = downloader.load_stock_list()

    # 开始下载
    downloader.download_all(
        stocks_df=stocks_df,
        resume=True
    )


if __name__ == "__main__":
    main()