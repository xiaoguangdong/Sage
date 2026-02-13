#!/usr/bin/env python3
"""
下载指数OHLCV数据

下载沪深300、中证500等指数的K线数据，用于计算市场特征
"""

import os
import sys
import pandas as pd
import logging
import time
from datetime import datetime
from pathlib import Path

import tushare as ts

from scripts.data._shared.runtime import get_tushare_token, setup_logger

logger = setup_logger(Path(__file__).stem)


class IndexOHLCDownloader:
    """指数OHLCV数据下载器"""
    
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
        self.output_dir = "data/tushare"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化Tushare Pro
        self.ts_pro = ts.pro_api(tushare_token)
        logger.info("Tushare Pro初始化成功")
        
        # API调用间隔（秒）
        self.delay = 2
    
    def download_index_data(self, index_code: str, index_name: str):
        """
        下载单个指数的OHLCV数据
        
        Args:
            index_code: 指数代码，如 '000300.SH'
            index_name: 指数名称，如 '沪深300'
            
        Returns:
            DataFrame包含指数数据
        """
        logger.info("=" * 70)
        logger.info(f"下载指数: {index_name} ({index_code})")
        logger.info("=" * 70)
        logger.info(f"  日期范围: {self.start_date} - {self.end_date}")
        
        try:
            # 调用daily接口获取指数数据
            df = self.ts_pro.daily(
                ts_code=index_code,
                start_date=self.start_date.replace('-', ''),
                end_date=self.end_date.replace('-', '')
            )
            
            if df is not None and not df.empty:
                logger.info(f"✓ 成功获取 {len(df)} 条记录")
                logger.info(f"  日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
                logger.info(f"  字段: {df.columns.tolist()}")
                
                # 重命名列以便后续使用
                df = df.rename(columns={
                    'trade_date': 'date',
                    'ts_code': 'code',
                    'pct_chg': 'pct_change'
                })
                
                # 保存数据
                output_path = os.path.join(
                    self.output_dir,
                    f"index_{index_code.replace('.', '_')}_ohlc.parquet"
                )
                df.to_parquet(output_path)
                logger.info(f"  保存路径: {output_path}")
                
                # 显示统计信息
                logger.info(f"\n统计信息:")
                logger.info(f"  收盘价范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
                logger.info(f"  平均成交量: {df['vol'].mean():,.0f}")
                
                return df
            else:
                logger.warning(f"✗ 数据为空")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"✗ 下载失败: {e}")
            return pd.DataFrame()
    
    def download_all_indices(self):
        """下载所有需要的指数数据"""
        logger.info("\n" + "=" * 70)
        logger.info("开始下载指数OHLCV数据")
        logger.info("=" * 70)
        
        # 定义需要下载的指数列表
        indices = [
            {
                'code': '000300.SH',
                'name': '沪深300',
                'priority': 1  # 高优先级，必需
            },
            {
                'code': '000905.SH',
                'name': '中证500',
                'priority': 2  # 中优先级
            }
        ]
        
        all_data = []
        success_count = 0
        failed_count = 0
        
        for idx_info in indices:
            index_code = idx_info['code']
            index_name = idx_info['name']
            priority = idx_info['priority']
            
            logger.info(f"\n优先级: {priority}")
            
            df = self.download_index_data(index_code, index_name)
            
            if not df.empty:
                df['index_name'] = index_name
                df['priority'] = priority
                all_data.append(df)
                success_count += 1
            else:
                failed_count += 1
            
            # API调用间隔
            time.sleep(self.delay)
        
        # 统计结果
        logger.info("\n" + "=" * 70)
        logger.info("下载完成！")
        logger.info("=" * 70)
        logger.info(f"  成功: {success_count}/{len(indices)} 个指数")
        logger.info(f"  失败: {failed_count}/{len(indices)} 个指数")
        
        if all_data:
            # 合并所有指数数据
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"  总记录数: {len(combined_df)}")
            
            # 保存合并数据
            output_path = os.path.join(self.output_dir, "index_ohlc_all.parquet")
            combined_df.to_parquet(output_path)
            logger.info(f"  合并数据保存: {output_path}")
            
            # 显示各指数统计
            logger.info("\n各指数统计:")
            for index_code in combined_df['code'].unique():
                df_idx = combined_df[combined_df['code'] == index_code]
                logger.info(f"  {index_code}:")
                logger.info(f"    记录数: {len(df_idx)}")
                logger.info(f"    日期范围: {df_idx['date'].min()} - {df_idx['date'].max()}")
                logger.info(f"    收盘价范围: {df_idx['close'].min():.2f} - {df_idx['close'].max():.2f}")
            
            return combined_df
        else:
            logger.warning("没有成功下载任何数据")
            return pd.DataFrame()


def main():
    """主函数"""
    # 配置
    TUSHARE_TOKEN = get_tushare_token()
    START_DATE = "2020-01-01"
    END_DATE = "2026-02-09"
    
    # 创建下载器
    downloader = IndexOHLCDownloader(
        tushare_token=TUSHARE_TOKEN,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # 开始下载
    downloader.download_all_indices()


if __name__ == "__main__":
    main()
