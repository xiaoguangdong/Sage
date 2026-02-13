#!/usr/bin/env python3
"""
获取上市公司主营业务分部数据（fina_mainbz_vip接口）
用于计算行业映射的真实权重

数据范围：2020-2026年
获取频率：每季度一次，间隔30-60秒
"""

import tushare as ts
import pandas as pd
import time
import os
from datetime import datetime
import logging

from tushare_auth import get_tushare_token
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinaMainbzFetcher:
    """主营业务分部数据获取器"""

    def __init__(self, token, batch_limit=8000):
        """
        初始化获取器

        Args:
            token: Tushare API token
            batch_limit: 每次分页获取的记录数限制
        """
        self.pro = ts.pro_api(token)
        self.batch_limit = batch_limit  # 每次获取8000条
        self.base_delay = 30  # 基础延迟30秒
        self.quarter_delay = 60  # 季度间延迟60秒

        # 输出目录
        self.output_dir = os.path.join("data", "tushare", "macro", "segments")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_all_quarters(self, start_year=2020, end_year=2026):
        """
        获取所有季度列表

        Args:
            start_year: 起始年份
            end_year: 结束年份

        Returns:
            季度列表，格式如 ['20201231', '20210331', ...]
        """
        quarters = []
        for year in range(start_year, end_year + 1):
            for quarter in ['0331', '0630', '0930', '1231']:
                quarter_str = f"{year}{quarter}"
                quarters.append(quarter_str)
        return quarters

    def fetch_mainbz_by_period(self, period, ts_code=None):
        """
        获取指定期间的主营业务分部数据（支持分页获取完整数据）

        Args:
            period: 报告期，格式如 '20231231'
            ts_code: 股票代码（可选），如果为None则获取所有股票

        Returns:
            DataFrame: 主营业务分部数据
        """
        logger.info(f"开始获取 {period} 的主营业务分部数据...")

        all_data = []
        offset = 0
        batch_num = 0

        try:
            while True:
                batch_num += 1
                logger.info(f"  获取第 {batch_num} 批数据 (offset={offset}, limit={self.batch_limit})...")

                # 调用 fina_mainbz_vip 接口，使用分页参数
                df = self.pro.fina_mainbz_vip(
                    period=period,
                    ts_code=ts_code,
                    offset=offset,
                    limit=self.batch_limit
                )

                if df is None or df.empty:
                    logger.info(f"  第 {batch_num} 批无数据，获取完成")
                    break

                all_data.append(df)
                logger.info(f"  第 {batch_num} 批获取成功，共 {len(df)} 条记录")

                # 如果返回的数据少于limit，说明已经获取完所有数据
                if len(df) < self.batch_limit:
                    break

                # 继续获取下一批
                offset += self.batch_limit
                # 批次之间短暂延迟
                time.sleep(0.5)

            # 合并所有批次的数据
            if all_data:
                result_df = pd.concat(all_data, ignore_index=True)
                logger.info(f"{period} 总共获取成功，共 {len(result_df)} 条记录 ({batch_num} 批)")
                return result_df
            else:
                logger.warning(f"{period} 没有数据")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"获取 {period} 数据失败: {e}")
            return pd.DataFrame()

    def save_to_parquet(self, df, period):
        """
        保存数据到parquet文件

        Args:
            df: DataFrame数据
            period: 报告期
        """
        if df is not None and not df.empty:
            filename = os.path.join(self.output_dir, f"fina_mainbz_{period}.parquet")
            df.to_parquet(filename, index=False)
            logger.info(f"数据已保存到: {filename}")

    def fetch_all_periods(self, start_year=2020, end_year=2026, ts_code=None):
        """
        获取所有期间的数据

        Args:
            start_year: 起始年份
            end_year: 结束年份
            ts_code: 股票代码（可选）
        """
        quarters = self.get_all_quarters(start_year, end_year)
        total = len(quarters)

        logger.info(f"准备获取 {total} 个季度的数据（{start_year}-{end_year}）")
        logger.info(f"预计耗时: {total * 60 / 60:.1f} 小时")

        for idx, period in enumerate(quarters, 1):
            logger.info(f"\n进度: {idx}/{total} ({idx/total*100:.1f}%)")

            # 获取数据
            df = self.fetch_mainbz_by_period(period, ts_code)

            # 保存数据
            self.save_to_parquet(df, period)

            # 如果不是最后一个季度，延迟
            if idx < total:
                delay = self.quarter_delay
                logger.info(f"等待 {delay} 秒后获取下一个季度...")
                time.sleep(delay)

        logger.info("\n所有数据获取完成！")


def main():
    """主函数"""
    token = get_tushare_token()

    # 创建获取器
    fetcher = FinaMainbzFetcher(token)

    # 获取所有数据（按季度下载全部股票）
    fetcher.fetch_all_periods(
        start_year=2020,
        end_year=2026,
        ts_code=None  # None表示获取所有股票
    )


if __name__ == "__main__":
    main()
