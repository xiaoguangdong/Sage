#!/usr/bin/env python3
"""
下载指数成分股和行业板块数据

功能:
1. 沪深300成分股列表和权重
2. 申万行业分类
3. 行业板块涨跌数据
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ml_stock_forecast.data.data_provider import DataProvider

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/tushare/index_data_download.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class IndexDataDownloader:
    """指数成分股和行业数据下载器"""
    
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
        
        # 初始化数据提供商
        self.provider = DataProvider(tushare_token=tushare_token)
    
    def download_hs300_constituents(self):
        """下载沪深300成分股"""
        logger.info("=" * 70)
        logger.info("开始下载沪深300成分股")
        logger.info("=" * 70)
        
        try:
            # 获取当前成分股
            df_constituents = self.provider.ts_pro.index_member(
                index_code='000300.SH'
            )
            
            if not df_constituents.empty:
                output_path = os.path.join(self.output_dir, "hs300_constituents.parquet")
                df_constituents.to_parquet(output_path)
                logger.info(f"✓ 沪深300成分股已保存: {len(df_constituents)} 只股票")
                logger.info(f"  保存路径: {output_path}")
                return df_constituents
            else:
                logger.warning("沪深300成分股数据为空")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"✗ 下载沪深300成分股失败: {e}")
            return pd.DataFrame()
    
    def download_hs300_weights(self):
        """下载沪深300成分股权重"""
        logger.info("=" * 70)
        logger.info("开始下载沪深300成分股权重")
        logger.info("=" * 70)
        
        try:
            # 获取权重数据
            df_weights = self.provider.ts_pro.index_weight(
                index_code='000300.SH',
                start_date=self.start_date.replace('-', ''),
                end_date=self.end_date.replace('-', '')
            )
            
            if not df_weights.empty:
                output_path = os.path.join(self.output_dir, "hs300_weights.parquet")
                df_weights.to_parquet(output_path)
                logger.info(f"✓ 沪深300权重数据已保存: {len(df_weights)} 条记录")
                logger.info(f"  日期范围: {df_weights['trade_date'].min()} - {df_weights['trade_date'].max()}")
                logger.info(f"  保存路径: {output_path}")
                return df_weights
            else:
                logger.warning("沪深300权重数据为空")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"✗ 下载沪深300权重数据失败: {e}")
            return pd.DataFrame()
    
    def download_sw_industry_classify(self):
        """下载申万行业分类"""
        logger.info("=" * 70)
        logger.info("开始下载申万行业分类")
        logger.info("=" * 70)
        
        try:
            # 获取申万一级行业分类
            df_industry_l1 = self.provider.ts_pro.index_classify(
                level='L1',
                src='SW'
            )
            
            # 获取申万二级行业分类
            df_industry_l2 = self.provider.ts_pro.index_classify(
                level='L2',
                src='SW'
            )
            
            # 获取申万三级行业分类
            df_industry_l3 = self.provider.ts_pro.index_classify(
                level='L3',
                src='SW'
            )
            
            if not df_industry_l1.empty:
                # 保存一级行业
                output_l1 = os.path.join(self.output_dir, "sw_industry_l1.parquet")
                df_industry_l1.to_parquet(output_l1)
                logger.info(f"✓ 申万一级行业: {len(df_industry_l1)} 个")
                
                # 保存二级行业
                if not df_industry_l2.empty:
                    output_l2 = os.path.join(self.output_dir, "sw_industry_l2.parquet")
                    df_industry_l2.to_parquet(output_l2)
                    logger.info(f"✓ 申万二级行业: {len(df_industry_l2)} 个")
                
                # 保存三级行业
                if not df_industry_l3.empty:
                    output_l3 = os.path.join(self.output_dir, "sw_industry_l3.parquet")
                    df_industry_l3.to_parquet(output_l3)
                    logger.info(f"✓ 申万三级行业: {len(df_industry_l3)} 个")
                
                return df_industry_l1
            else:
                logger.warning("申万行业分类数据为空")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"✗ 下载申万行业分类失败: {e}")
            return pd.DataFrame()
    
    def download_industry_daily_data(self):
        """下载行业指数日线数据"""
        logger.info("=" * 70)
        logger.info("开始下载行业指数日线数据")
        logger.info("=" * 70)
        
        try:
            # 先获取行业列表
            df_industry_l1 = pd.read_parquet(os.path.join(self.output_dir, "sw_industry_l1.parquet"))
            
            if df_industry_l1.empty:
                logger.warning("行业分类数据为空，跳过下载")
                return pd.DataFrame()
            
            # 下载每个行业的日线数据
            all_industry_data = []
            
            for idx, row in df_industry_l1.iterrows():
                industry_code = row['index_code']
                industry_name = row['industry_name']
                
                logger.info(f"  下载行业: {industry_name} ({industry_code})")
                
                try:
                    df_daily = self.provider.ts_pro.daily(
                        ts_code=industry_code,
                        start_date=self.start_date.replace('-', ''),
                        end_date=self.end_date.replace('-', '')
                    )
                    
                    if not df_daily.empty:
                        df_daily['industry_name'] = industry_name
                        all_industry_data.append(df_daily)
                        logger.info(f"    ✓ {len(df_daily)} 条记录")
                    else:
                        logger.warning(f"    ✗ 数据为空")
                        
                except Exception as e:
                    logger.error(f"    ✗ 下载失败: {e}")
                    continue
            
            # 合并所有行业数据
            if all_industry_data:
                combined_df = pd.concat(all_industry_data, ignore_index=True)
                output_path = os.path.join(self.output_dir, "sw_industry_daily.parquet")
                combined_df.to_parquet(output_path)
                logger.info(f"\n✓ 行业指数日线数据已保存: {len(combined_df)} 条记录")
                logger.info(f"  保存路径: {output_path}")
                return combined_df
            else:
                logger.warning("没有成功下载任何行业数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"✗ 下载行业指数日线数据失败: {e}")
            return pd.DataFrame()
    
    def download_all(self):
        """下载所有数据"""
        logger.info("\n" + "=" * 70)
        logger.info("开始下载指数成分股和行业数据")
        logger.info("=" * 70)
        
        # 1. 下载沪深300成分股
        self.download_hs300_constituents()
        
        # 2. 下载沪深300权重
        self.download_hs300_weights()
        
        # 3. 下载申万行业分类
        self.download_sw_industry_classify()
        
        # 4. 下载行业指数日线数据
        self.download_industry_daily_data()
        
        logger.info("\n" + "=" * 70)
        logger.info("所有数据下载完成！")
        logger.info("=" * 70)


def main():
    """主函数"""
    # 配置
    TUSHARE_TOKEN = "2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e"
    START_DATE = "2020-01-01"
    END_DATE = "2026-02-09"
    
    # 创建下载器
    downloader = IndexDataDownloader(
        tushare_token=TUSHARE_TOKEN,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # 开始下载
    downloader.download_all()


if __name__ == "__main__":
    main()