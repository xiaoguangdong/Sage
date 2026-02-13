#!/usr/bin/env python3
"""
下载期权日线数据（opt_daily）
"""
import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta

import tushare as ts

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/data/opt_daily_download.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class OptDailyDownloader:
    """期权日线数据下载器"""
    
    def __init__(self, tushare_token: str, start_date: str = "2020-01-01", end_date: str = "2026-02-09"):
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
        self.output_dir = "data/tushare/options"
        
        # API调用间隔（秒）
        self.api_delay = 1  # 期权数据量不大，1秒即可
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化Tushare Pro
        self.ts_pro = ts.pro_api(tushare_token)
        logger.info("Tushare Pro初始化成功")
    
    def generate_dates(self):
        """生成交易日列表"""
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        
        logger.info(f"生成日期列表: {len(dates)} 个日期")
        return dates
    
    def download_daily_for_date(self, trade_date: str, exchange: str = 'SSE'):
        """
        下载指定交易日的期权数据
        
        Args:
            trade_date: 交易日，格式YYYYMMDD
            exchange: 交易所（SSE上交所 / SZSE深交所）
            
        Returns:
            DataFrame包含该日所有期权数据
        """
        try:
            df = self.ts_pro.opt_daily(
                trade_date=trade_date,
                exchange=exchange,
                calender='natural'
            )
            
            if df is not None and not df.empty:
                logger.info(f"{exchange} {trade_date} 下载了 {len(df)} 条记录")
                return df
            else:
                logger.info(f"{exchange} {trade_date} 无数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"{exchange} {trade_date} 下载失败: {e}")
            return pd.DataFrame()
    
    def download_all(self, exchanges=['SSE', 'SZSE']):
        """
        下载所有日期的期权数据
        
        Args:
            exchanges: 交易所列表
        """
        logger.info("\n" + "=" * 70)
        logger.info("开始下载期权日线数据")
        logger.info("=" * 70)
        logger.info(f"  日期范围: {self.start_date} - {self.end_date}")
        logger.info(f"  交易所: {', '.join(exchanges)}")
        logger.info("=" * 70)
        
        # 生成日期列表
        dates = self.generate_dates()
        
        # 下载每个交易所的数据
        all_data = {}
        for exchange in exchanges:
            logger.info(f"\n开始下载 {exchange} 期权数据...")
            exchange_data = []
            
            for i, date in enumerate(dates, 1):
                logger.info(f"\n进度: {i}/{len(dates)}")
                
                df = self.download_daily_for_date(date, exchange)
                
                if not df.empty:
                    exchange_data.append(df)
                
                # API调用间隔
                import time
                time.sleep(self.api_delay)
            
            # 合并该交易所所有数据
            if exchange_data:
                combined_df = pd.concat(exchange_data, ignore_index=True)
                all_data[exchange] = combined_df
                
                # 保存该交易所数据
                output_path = os.path.join(self.output_dir, f"opt_daily_{exchange.lower()}.parquet")
                combined_df.to_parquet(output_path)
                logger.info(f"\n{exchange} 数据保存: {output_path}")
                logger.info(f"  总记录数: {len(combined_df)}")
                logger.info(f"  日期范围: {combined_df['trade_date'].min()} - {combined_df['trade_date'].max()}")
                logger.info(f"  期权数量: {combined_df['ts_code'].nunique()}")
            else:
                logger.warning(f"\n{exchange} 没有成功下载任何数据")
        
        # 统计结果
        logger.info("\n" + "=" * 70)
        logger.info("下载完成！")
        logger.info("=" * 70)
        
        for exchange, df in all_data.items():
            logger.info(f"\n{exchange}:")
            logger.info(f"  总记录数: {len(df)}")
            logger.info(f"  期权数量: {df['ts_code'].nunique()}")
            logger.info(f"  日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
        
        return all_data


def main():
    """主函数"""
    # 配置
    TUSHARE_TOKEN = "2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e"
    START_DATE = "2020-01-01"
    END_DATE = "2026-02-09"
    
    # 创建下载器
    downloader = OptDailyDownloader(
        tushare_token=TUSHARE_TOKEN,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # 开始下载（上交所和深交所）
    downloader.download_all(exchanges=['SSE', 'SZSE'])


if __name__ == "__main__":
    main()