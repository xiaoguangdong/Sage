#!/usr/bin/env python3
"""
重新获取半年报和年报的主营业务分部数据
使用分页获取，避免10,000条限制

数据范围：2020-2026年的半年报(0630)和年报(1231)
"""

import tushare as ts
import pandas as pd
import time
import os
import logging

from tushare_auth import get_tushare_token
from scripts.data.macro.paths import MACRO_DIR
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HalfYearReportFetcher:
    """半年报和年报数据重新获取器"""

    def __init__(self, token, batch_limit=8000, api_delay=45):
        """
        初始化获取器

        Args:
            token: Tushare API token
            batch_limit: 每次分页获取的记录数限制
            api_delay: 每次API请求之间的延迟（秒）
        """
        self.pro = ts.pro_api(token)
        self.batch_limit = batch_limit
        self.api_delay = api_delay
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 60  # 重试延迟60秒
        self.output_dir = str(MACRO_DIR / "segments")
        os.makedirs(self.output_dir, exist_ok=True)

    def get_half_year_quarters(self, start_year=2020, end_year=2026):
        """获取所有半年报和年报的季度列表"""
        quarters = []
        for year in range(start_year, end_year + 1):
            # 只添加半年报(0630)和年报(1231)
            quarters.extend([f"{year}0630", f"{year}1231"])
        return quarters

    def fetch_mainbz_by_period(self, period):
        """
        获取指定期间的主营业务分部数据（分页获取）

        Args:
            period: 报告期，格式如 '20231231'

        Returns:
            DataFrame: 主营业务分部数据
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始获取 {period} 的主营业务分部数据...")
        logger.info(f"{'='*60}")

        all_data = []
        offset = 0
        batch_num = 0

        try:
            while True:
                batch_num += 1
                retry_count = 0
                
                while retry_count < self.max_retries:
                    try:
                        logger.info(f"[{period}] 第 {batch_num} 批: offset={offset}, limit={self.batch_limit} (重试 {retry_count + 1}/{self.max_retries})")

                        # 调用 fina_mainbz_vip 接口
                        df = self.pro.fina_mainbz_vip(
                            period=period,
                            offset=offset,
                            limit=self.batch_limit
                        )

                        if df is None or df.empty:
                            logger.info(f"[{period}] 第 {batch_num} 批无数据，获取完成")
                            return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

                        all_data.append(df)
                        logger.info(f"[{period}] 第 {batch_num} 批获取成功，共 {len(df)} 条记录")

                        # 成功获取，跳出重试循环
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        error_msg = str(e)
                        
                        if "IP数量超限" in error_msg or "超过2个" in error_msg:
                            logger.warning(f"[{period}] ⚠️ IP限制触发，等待{self.retry_delay}秒后重试...")
                            time.sleep(self.retry_delay)
                        else:
                            logger.warning(f"[{period}] ⚠️ 获取失败: {e}，等待10秒后重试...")
                            time.sleep(10)
                        
                        # 如果达到最大重试次数
                        if retry_count >= self.max_retries:
                            logger.error(f"[{period}] ❌ 达到最大重试次数，获取失败")
                            # 保存已获取的数据
                            if all_data:
                                result_df = pd.concat(all_data, ignore_index=True)
                                filename = os.path.join(self.output_dir, f"fina_mainbz_{period}_partial.parquet")
                                result_df.to_parquet(filename, index=False)
                                logger.warning(f"[{period}] 💾 部分数据已保存到: {filename}")
                            return pd.DataFrame()

                # 如果返回的数据少于limit，说明已经获取完所有数据
                if len(df) < self.batch_limit:
                    break

                # 继续获取下一批
                offset += self.batch_limit
                # 批次之间延迟
                time.sleep(self.api_delay)

            # 合并所有批次的数据
            if all_data:
                result_df = pd.concat(all_data, ignore_index=True)
                logger.info(f"[{period}] ✅ 总共获取 {len(result_df):,} 条记录 ({batch_num} 批)")

                # 保存数据
                filename = os.path.join(self.output_dir, f"fina_mainbz_{period}.parquet")
                result_df.to_parquet(filename, index=False)
                logger.info(f"[{period}] 💾 数据已保存到: {filename}")

                return result_df
            else:
                logger.warning(f"[{period}] ⚠️ 没有数据")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"[{period}] ❌ 获取失败: {e}")
            return pd.DataFrame()

    def fetch_all_half_year_reports(self, start_year=2020, end_year=2026):
        """
        获取所有半年报和年报数据

        Args:
            start_year: 起始年份
            end_year: 结束年份
        """
        quarters = self.get_half_year_quarters(start_year, end_year)
        total = len(quarters)

        logger.info(f"📊 准备重新获取 {total} 个半年报和年报数据（{start_year}-{end_year}）")
        logger.info(f"⏱️ 预计耗时: 约 {total * 2} 分钟")

        total_records = 0

        for idx, period in enumerate(quarters, 1):
            logger.info(f"\n{'#'*60}")
            logger.info(f"进度: {idx}/{total} ({idx/total*100:.1f}%)")
            logger.info(f"{'#'*60}")

            # 获取数据
            df = self.fetch_mainbz_by_period(period)

            if not df.empty:
                total_records += len(df)

            # 季度之间延迟60秒（避免IP限制）
            if idx < total:
                logger.info(f"\n⏳ 等待 60 秒后获取下一个季度...")
                time.sleep(60)

        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 所有数据获取完成！")
        logger.info(f"📈 总计获取: {total_records:,} 条记录")
        logger.info(f"📁 数据保存在: {self.output_dir}")
        logger.info(f"{'='*60}")


def main():
    """主函数"""
    # Tushare token
    TUSHARE_TOKEN = get_tushare_token()

    # 创建获取器（batch_limit=8000, api_delay=45）
    fetcher = HalfYearReportFetcher(token=TUSHARE_TOKEN, batch_limit=8000, api_delay=45)

    # 获取所有半年报和年报数据
    fetcher.fetch_all_half_year_reports(
        start_year=2020,
        end_year=2026
    )


if __name__ == "__main__":
    main()
