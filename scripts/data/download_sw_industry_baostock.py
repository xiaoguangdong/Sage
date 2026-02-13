#!/usr/bin/env python3
"""
使用Baostock下载申万行业指数数据

Baostock的优势：
1. 不受Tushare IP限制影响
2. 申万行业数据完整
3. API调用限制较宽松
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime

import baostock as bs

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/tushare/sw_industry_baostock.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class BaostockIndustryDownloader:
    """Baostock行业数据下载器"""
    
    def __init__(self, start_date: str = "2020-01-01", end_date: str = "2026-02-09"):
        """
        初始化下载器
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        """
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = "data/tushare"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 登录Baostock
        lg = bs.login()
        if lg.error_code != '0':
            raise Exception(f"Baostock登录失败: {lg.error_msg}")
        
        logger.info("Baostock登录成功")
    
    def download_sw_industry_list(self):
        """获取申万行业列表"""
        logger.info("=" * 70)
        logger.info("开始获取申万行业列表")
        logger.info("=" * 70)
        
        try:
            # 获取申万一级行业列表
            rs = bs.query_stock_industry()
            
            if rs.error_code != '0':
                logger.error(f"获取行业列表失败: {rs.error_msg}")
                return pd.DataFrame()
            
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            if not df.empty:
                output_path = os.path.join(self.output_dir, "sw_industry_list.parquet")
                df.to_parquet(output_path)
                logger.info(f"✓ 申万行业列表已保存: {len(df)} 个行业")
                logger.info(f"  保存路径: {output_path}")
                
                # 显示申万一级行业
                sw_l1 = df[df['level'] == '1.0']
                logger.info(f"  申万一级行业: {len(sw_l1)} 个")
                for _, row in sw_l1.head(10).iterrows():
                    logger.info(f"    {row['industry_name']} ({row['industry']})")
                
                return df
            else:
                logger.warning("行业列表为空")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"✗ 获取行业列表失败: {e}")
            return pd.DataFrame()
    
    def download_sw_industry_daily(self):
        """下载申万行业指数日线数据"""
        logger.info("=" * 70)
        logger.info("开始下载申万行业指数日线数据")
        logger.info("=" * 70)
        
        try:
            # 先获取行业列表
            df_industry = pd.read_parquet(os.path.join(self.output_dir, "sw_industry_list.parquet"))
            
            if df_industry.empty:
                logger.warning("行业列表为空，跳过下载")
                return pd.DataFrame()
            
            # 筛选申万一级行业
            sw_l1 = df_industry[df_industry['level'] == '1.0']
            logger.info(f"申万一级行业数量: {len(sw_l1)}")
            
            # 下载每个行业的日线数据
            all_industry_data = []
            
            for idx, row in sw_l1.iterrows():
                industry_code = row['industry']
                industry_name = row['industry_name']
                
                logger.info(f"  下载行业: {industry_name} ({industry_code})")
                
                try:
                    rs = bs.query_history_k_data_plus(
                        industry_code,
                        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                        start_date=self.start_date,
                        end_date=self.end_date,
                        frequency="d",
                        adjustflag="2"  # 后复权
                    )
                    
                    if rs.error_code != '0':
                        logger.warning(f"    ✗ 获取失败: {rs.error_msg}")
                        continue
                    
                    data_list = []
                    while (rs.error_code == '0') & rs.next():
                        data_list.append(rs.get_row_data())
                    
                    if data_list:
                        df_daily = pd.DataFrame(data_list, columns=rs.fields)
                        df_daily['industry_name'] = industry_name
                        df_daily['industry_code'] = industry_code
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
                logger.info(f"\n✓ 申万行业指数日线数据已保存: {len(combined_df)} 条记录")
                logger.info(f"  日期范围: {combined_df['date'].min()} - {combined_df['date'].max()}")
                logger.info(f"  保存路径: {output_path}")
                return combined_df
            else:
                logger.warning("没有成功下载任何行业数据")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"✗ 下载行业指数日线数据失败: {e}")
            return pd.DataFrame()
    
    def logout(self):
        """登出Baostock"""
        bs.logout()
        logger.info("Baostock已登出")
    
    def download_all(self):
        """下载所有数据"""
        logger.info("\n" + "=" * 70)
        logger.info("开始下载Baostock申万行业数据")
        logger.info("=" * 70)
        
        # 1. 获取行业列表
        self.download_sw_industry_list()
        
        # 2. 下载行业指数日线数据
        self.download_sw_industry_daily()
        
        # 3. 登出
        self.logout()
        
        logger.info("\n" + "=" * 70)
        logger.info("所有数据下载完成！")
        logger.info("=" * 70)


def main():
    """主函数"""
    # 配置
    START_DATE = "2020-01-01"
    END_DATE = "2026-02-09"
    
    # 创建下载器
    downloader = BaostockIndustryDownloader(
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # 开始下载
    downloader.download_all()


if __name__ == "__main__":
    main()