#!/usr/bin/env python3
"""
Tushare K线数据下载脚本

功能：
1. 下载Tushare日线数据（K线数据）
2. 按年份保存数据
3. 支持断点续传
4. 按天拉取，每次延时2秒
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta
import time
import json
from pathlib import Path

import tushare as ts

from scripts.data._shared.runtime import get_tushare_token, setup_logger

logger = setup_logger(Path(__file__).stem)


class DailyKlineDownloader:
    """Tushare日线K线数据下载器"""
    
    def __init__(self, tushare_token: str, start_date: str = "2020-01-01", end_date: str = "2026-02-06"):
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
        self.output_dir = "data/tushare/daily"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化Tushare Pro
        self.ts_pro = ts.pro_api(tushare_token)
        logger.info("Tushare Pro初始化成功")
        
        # 状态文件
        self.state_file = os.path.join(self.output_dir, "download_state.json")
        
        # 延时配置（秒）
        self.delay = 2  # 每次API调用之间延时2秒
    
    def generate_daily_ranges(self, start_date: str, end_date: str) -> list:
        """
        生成按天划分的日期列表
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            日期列表，每个元素是日期字符串（格式：YYYYMMDD）
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        dates = []
        current = start
        
        while current <= end:
            dates.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)
        
        return dates
    
    def load_state(self) -> dict:
        """加载下载状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logger.info(f"加载状态: 已完成{len(state.get('completed_dates', []))}个交易日")
                return state
            except Exception as e:
                logger.warning(f"加载状态文件失败: {e}")
        return {"completed_dates": [], "failed_dates": [], "total_records": 0}
    
    def save_state(self, state: dict):
        """保存下载状态"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存状态文件失败: {e}")
    
    def get_trading_dates(self, start_date: str, end_date: str) -> list:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            
        Returns:
            交易日列表
        """
        try:
            df = self.ts_pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
            trading_dates = df[df['is_open'] == 1]['cal_date'].tolist()
            return trading_dates
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []
    
    def download_daily_for_date(self, trade_date: str) -> pd.DataFrame:
        """
        下载指定交易日的日线数据
        
        Args:
            trade_date: 交易日期 (YYYYMMDD)
            
        Returns:
            日线数据DataFrame
        """
        try:
            # 使用daily接口获取日线数据（按天拉取）
            df = self.ts_pro.daily(trade_date=trade_date)
            
            if df is None or df.empty:
                logger.warning(f"{trade_date} 无数据")
                return pd.DataFrame()
            
            logger.info(f"{trade_date} 下载了 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"{trade_date} 下载失败: {e}")
            return pd.DataFrame()
    
    def save_yearly_data(self, df: pd.DataFrame, year: int):
        """
        按年保存数据
        
        Args:
            df: 数据DataFrame
            year: 年份
        """
        output_path = os.path.join(self.output_dir, f"daily_{year}.parquet")
        
        try:
            # 如果文件已存在，读取并追加
            if os.path.exists(output_path):
                existing_df = pd.read_parquet(output_path)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            # 去重
            df = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last')
            
            # 保存
            df.to_parquet(output_path, index=False)
            logger.info(f"已保存 {year} 年数据: {len(df)} 条记录")
        except Exception as e:
            logger.error(f"保存 {year} 年数据失败: {e}")
    
    def run(self):
        """执行下载"""
        logger.info("="*70)
        logger.info("开始下载Tushare日线K线数据")
        logger.info(f"时间范围: {self.start_date} ~ {self.end_date}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"每次API调用延时: {self.delay}秒")
        logger.info("="*70)
        
        # 加载状态
        state = self.load_state()
        completed_dates = set(state.get('completed_dates', []))
        failed_dates = set(state.get('failed_dates', []))
        
        # 生成所有日期
        all_dates = self.generate_daily_ranges(self.start_date, self.end_date)
        
        logger.info(f"共 {len(all_dates)} 个日期")
        logger.info(f"已完成 {len(completed_dates)} 个日期")
        logger.info(f"待下载 {len(all_dates) - len(completed_dates)} 个日期")
        
        # 按年份组织数据
        yearly_data = {}
        
        # 逐天下载
        success_count = 0
        fail_count = 0
        
        for i, trade_date in enumerate(all_dates):
            # 跳过已完成的日期
            if trade_date in completed_dates:
                continue
            
            logger.info(f"\n[{i+1}/{len(all_dates)}] 处理交易日: {trade_date}")
            
            # 延时
            time.sleep(self.delay)
            
            # 下载数据
            df = self.download_daily_for_date(trade_date)
            
            if df.empty:
                failed_dates.add(trade_date)
                fail_count += 1
                continue
            
            # 按年份组织数据
            year = int(trade_date[:4])
            if year not in yearly_data:
                yearly_data[year] = []
            yearly_data[year].append(df)
            
            # 标记完成
            completed_dates.add(trade_date)
            if trade_date in failed_dates:
                failed_dates.remove(trade_date)
            
            success_count += 1
            
            # 每下载100个交易日保存一次
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i+1} 个交易日，保存数据...")
                for y, data_list in yearly_data.items():
                    if data_list:
                        year_df = pd.concat(data_list, ignore_index=True)
                        self.save_yearly_data(year_df, y)
                        yearly_data[y] = []  # 清空已保存的数据
                
                # 保存状态
                state['completed_dates'] = sorted(list(completed_dates))
                state['failed_dates'] = sorted(list(failed_dates))
                self.save_state(state)
        
        # 保存剩余数据
        logger.info("\n保存剩余数据...")
        for year, data_list in yearly_data.items():
            if data_list:
                year_df = pd.concat(data_list, ignore_index=True)
                self.save_yearly_data(year_df, year)
        
        # 保存最终状态
        state['completed_dates'] = sorted(list(completed_dates))
        state['failed_dates'] = sorted(list(failed_dates))
        self.save_state(state)
        
        logger.info("="*70)
        logger.info("下载完成!")
        logger.info(f"成功: {success_count} 个交易日")
        logger.info(f"失败: {fail_count} 个交易日")
        logger.info(f"数据保存位置: {self.output_dir}")
        logger.info("="*70)


if __name__ == '__main__':
    # Tushare token
    TUSHARE_TOKEN = get_tushare_token()
    
    # 创建下载器
    downloader = DailyKlineDownloader(
        tushare_token=TUSHARE_TOKEN,
        start_date="2020-01-01",
        end_date="2026-02-06"
    )
    
    # 执行下载
    downloader.run()
