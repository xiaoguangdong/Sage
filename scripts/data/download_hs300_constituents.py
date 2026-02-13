#!/usr/bin/env python3
"""
下载沪深300成分股列表（2020年以来）

使用 index_weight 接口按年份分批下载
"""
import os
import sys
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path

import tushare as ts

from scripts.data._shared.runtime import get_tushare_token, setup_logger

logger = setup_logger(Path(__file__).stem)


class HS300ConstituentsDownloader:
    """沪深300成分股下载器"""
    
    def __init__(self, tushare_token: str, start_year: int = 2020, end_year: int = 2026):
        """
        初始化下载器
        
        Args:
            tushare_token: Tushare token
            start_year: 起始年份
            end_year: 结束年份
        """
        self.tushare_token = tushare_token
        self.start_year = start_year
        self.end_year = end_year
        self.output_dir = "data/tushare"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化Tushare Pro
        self.ts_pro = ts.pro_api(tushare_token)
        logger.info("Tushare Pro初始化成功")
        
        # API调用间隔（秒）
        self.delay = 1
    
    def download_year(self, year: int) -> pd.DataFrame:
        """
        下载指定年份的成分股数据
        
        Args:
            year: 年份
            
        Returns:
            DataFrame包含该年的成分股数据
        """
        logger.info("=" * 70)
        logger.info(f"下载 {year} 年沪深300成分股")
        logger.info("=" * 70)
        
        try:
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            # 调用index_weight接口
            df = self.ts_pro.index_weight(
                index_code='000300.SH',
                start_date=start_date,
                end_date=end_date
            )
            
            if df is not None and not df.empty:
                logger.info(f"✓ 成功获取 {len(df)} 条记录")
                logger.info(f"  日期范围: {df['trade_date'].min()} - {df['trade_date'].max()}")
                logger.info(f"  涉及股票数: {df['con_code'].nunique()}")
                
                # 保存年度数据
                output_path = os.path.join(
                    self.output_dir,
                    f"hs300_constituents_{year}.parquet"
                )
                df.to_parquet(output_path)
                logger.info(f"  保存路径: {output_path}")
                
                return df
            else:
                logger.warning(f"✗ 数据为空")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"✗ 下载失败: {e}")
            return pd.DataFrame()
    
    def download_all(self):
        """下载所有年份的成分股数据"""
        logger.info("\n" + "=" * 70)
        logger.info("开始下载沪深300成分股数据（按年份）")
        logger.info("=" * 70)
        logger.info(f"  年份范围: {self.start_year} - {self.end_year}")
        logger.info("=" * 70)
        
        all_data = []
        success_count = 0
        failed_count = 0
        
        for year in range(self.start_year, self.end_year + 1):
            logger.info(f"\n进度: {year - self.start_year + 1}/{self.end_year - self.start_year + 1}")
            
            df = self.download_year(year)
            
            if not df.empty:
                all_data.append(df)
                success_count += 1
            else:
                failed_count += 1
        
        # 统计结果
        logger.info("\n" + "=" * 70)
        logger.info("下载完成！")
        logger.info("=" * 70)
        logger.info(f"  成功: {success_count}/{self.end_year - self.start_year + 1} 年")
        logger.info(f"  失败: {failed_count}/{self.end_year - self.start_year + 1} 年")
        
        if all_data:
            # 合并所有年份数据
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"  总记录数: {len(combined_df)}")
            logger.info(f"  总股票数: {combined_df['con_code'].nunique()}")
            logger.info(f"  日期范围: {combined_df['trade_date'].min()} - {combined_df['trade_date'].max()}")
            
            # 保存合并数据
            output_path = os.path.join(self.output_dir, "hs300_constituents_all.parquet")
            combined_df.to_parquet(output_path)
            logger.info(f"  合并数据保存: {output_path}")
            
            # 最新成分股
            latest_date = combined_df['trade_date'].max()
            latest_constituents = combined_df[combined_df['trade_date'] == latest_date]
            logger.info(f"\n最新成分股（{latest_date}）:")
            logger.info(f"  数量: {len(latest_constituents)}")
            logger.info(f"  Top 10权重:")
            top10 = latest_constituents.nlargest(10, 'weight')
            logger.info(top10[['con_code', 'weight']].to_string(index=False))
            
            return combined_df
        else:
            logger.warning("没有成功下载任何数据")
            return pd.DataFrame()


def main():
    """主函数"""
    # 配置
    TUSHARE_TOKEN = get_tushare_token()
    START_YEAR = 2020
    END_YEAR = 2026
    
    # 创建下载器
    downloader = HS300ConstituentsDownloader(
        tushare_token=TUSHARE_TOKEN,
        start_year=START_YEAR,
        end_year=END_YEAR
    )
    
    # 开始下载
    downloader.download_all()


if __name__ == "__main__":
    main()
