#!/usr/bin/env python3
"""
使用fina_indicator_vip接口按季度下载财务指标数据

优势：
1. 一次性获取某个季度所有上市公司的数据
2. 不需要逐个股票调用API
3. 大幅提升下载速度

限制：
- 需要5000积分
- 按季度获取，不是按日期范围
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


class FinaIndicatorVIPDownloader:
    """财务指标VIP接口下载器"""
    
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
    
    def generate_quarter_periods(self):
        """生成季度列表"""
        quarters = []
        
        for year in range(self.start_year, self.end_year + 1):
            # 每年4个季度
            quarters.append(f"{year}0331")  # Q1
            quarters.append(f"{year}0630")  # Q2
            quarters.append(f"{year}0930")  # Q3
            quarters.append(f"{year}1231")  # Q4
        
        logger.info(f"生成季度列表: {len(quarters)} 个季度")
        logger.info(f"  起始: {quarters[0]}, 结束: {quarters[-1]}")
        
        return quarters
    
    def download_quarter(self, period: str):
        """
        下载单个季度的财务数据
        
        Args:
            period: 报告期，如 20231231
            
        Returns:
            DataFrame包含该季度所有公司的财务数据
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"下载季度: {period}")
        logger.info(f"{'='*70}")
        
        try:
            # 调用fina_indicator_vip接口
            df = self.ts_pro.fina_indicator_vip(
                period=period
            )
            
            if df is not None and not df.empty:
                logger.info(f"✓ 成功获取 {len(df)} 条记录")
                logger.info(f"  涉及股票数: {df['ts_code'].nunique()}")
                logger.info(f"  列数: {len(df.columns)}")
                
                # 保存数据
                output_path = os.path.join(
                    self.output_dir,
                    f"fina_indicator_vip_{period}.parquet"
                )
                df.to_parquet(output_path)
                logger.info(f"  保存路径: {output_path}")
                
                # 显示前几列
                logger.info(f"  主要字段: {df.columns[:10].tolist()}")
                
                return df
            else:
                logger.warning(f"✗ 数据为空")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"✗ 下载失败: {e}")
            return pd.DataFrame()
    
    def download_all(self):
        """下载所有季度的财务数据"""
        logger.info("\n" + "=" * 70)
        logger.info("开始下载财务指标数据（VIP接口，按季度）")
        logger.info("=" * 70)
        logger.info(f"  年份范围: {self.start_year} - {self.end_year}")
        logger.info(f"  接口: fina_indicator_vip")
        logger.info("=" * 70)
        
        # 生成季度列表
        quarters = self.generate_quarter_periods()
        
        # 下载每个季度
        all_data = []
        success_count = 0
        failed_count = 0
        
        for i, period in enumerate(quarters, 1):
            logger.info(f"\n进度: {i}/{len(quarters)}")
            
            df = self.download_quarter(period)
            
            if not df.empty:
                all_data.append(df)
                success_count += 1
            else:
                failed_count += 1
        
        # 统计结果
        logger.info("\n" + "=" * 70)
        logger.info("下载完成！")
        logger.info("=" * 70)
        logger.info(f"  成功: {success_count}/{len(quarters)} 个季度")
        logger.info(f"  失败: {failed_count}/{len(quarters)} 个季度")
        
        if all_data:
            # 合并所有季度数据
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"  总记录数: {len(combined_df)}")
            logger.info(f"  总股票数: {combined_df['ts_code'].nunique()}")
            
            # 保存合并数据
            output_path = os.path.join(self.output_dir, "fina_indicator_vip_all.parquet")
            combined_df.to_parquet(output_path)
            logger.info(f"  合并数据保存: {output_path}")
            
            # 按季度统计
            quarterly_stats = combined_df.groupby('end_date').agg({
                'ts_code': 'nunique',
                'report_type': 'count'
            }).rename(columns={'ts_code': '股票数', 'report_type': '报告数'})
            logger.info("\n各季度统计:")
            logger.info(quarterly_stats.to_string())
            
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
    downloader = FinaIndicatorVIPDownloader(
        tushare_token=TUSHARE_TOKEN,
        start_year=START_YEAR,
        end_year=END_YEAR
    )
    
    # 开始下载
    downloader.download_all()


if __name__ == "__main__":
    main()
