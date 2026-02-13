#!/usr/bin/env python3
"""
优化版daily_basic下载脚本

优化策略：
1. 按月下载，失败时按半月重试
2. 半月的数据量不会超过8万条
3. 识别已下载的数据，避免重复下载
4. 断点续传支持
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path

import tushare as ts

from scripts.data._shared.runtime import get_tushare_token, setup_logger

logger = setup_logger(Path(__file__).stem)


class DailyBasicOptimizedDownloader:
    """优化版daily_basic下载器"""
    
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
        
        # 安全配置
        self.max_records_per_call = 80000  # 每次最多8万条记录
        
        # 状态文件
        self.state_file = os.path.join(self.output_dir, "daily_basic_state_optimized.json")
        
        # 加载已下载数据的日期
        self.downloaded_dates = self._load_downloaded_dates()
    
    def _load_downloaded_dates(self) -> set:
        """加载已下载的日期"""
        try:
            output_path = os.path.join(self.output_dir, "daily_basic_all.parquet")
            if os.path.exists(output_path):
                df = pd.read_parquet(output_path)
                dates = set(df['trade_date'].unique())
                logger.info(f"已下载数据: {len(dates)} 个交易日, {len(df):,} 条记录")
                return dates
        except Exception as e:
            logger.warning(f"读取已下载数据失败: {e}")
        return set()
    
    def generate_month_ranges(self, start_date: str, end_date: str):
        """
        生成按月份划分的日期范围列表
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            日期范围列表，每个元素是 (start, end) 元组（格式：YYYYMMDD）
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        ranges = []
        current = start
        
        while current <= end:
            # 计算当前月份的最后一天
            if current.month == 12:
                next_month = current.replace(year=current.year + 1, month=1, day=1)
            else:
                next_month = current.replace(month=current.month + 1, day=1)
            
            # 当前月份的最后一天是下个月1号的前一天
            month_end = next_month - timedelta(days=1)
            
            # 确保不超过结束日期
            if month_end > end:
                month_end = end
            
            # 添加日期范围
            ranges.append((
                current.strftime('%Y%m%d'),
                month_end.strftime('%Y%m%d')
            ))
            
            # 移动到下个月
            current = next_month
        
        return ranges
    
    def split_month_to_half_months(self, month_start: str, month_end: str):
        """
        将月份拆分为两个半月
        
        Args:
            month_start: 月份开始日期 (YYYYMMDD)
            month_end: 月份结束日期 (YYYYMMDD)
            
        Returns:
            两个半月的日期范围列表
        """
        start_date = datetime.strptime(month_start, '%Y%m%d')
        end_date = datetime.strptime(month_end, '%Y%m%d')
        
        # 计算中间日期
        days_in_month = (end_date - start_date).days + 1
        mid_day = start_date + timedelta(days=days_in_month // 2)
        
        # 第一个半月
        half1_start = start_date.strftime('%Y%m%d')
        half1_end = (mid_day - timedelta(days=1)).strftime('%Y%m%d')
        
        # 第二个半月
        half2_start = mid_day.strftime('%Y%m%d')
        half2_end = end_date.strftime('%Y%m%d')
        
        return [
            (half1_start, half1_end),
            (half2_start, half2_end)
        ]
    
    def get_all_dates_in_range(self, range_start: str, range_end: str) -> list:
        """获取日期范围内的所有日期"""
        start = datetime.strptime(range_start, '%Y%m%d')
        end = datetime.strptime(range_end, '%Y%m%d')
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y%m%d'))
            current = current + timedelta(days=1)
        
        return dates
    
    def is_range_downloaded(self, range_start: str, range_end: str) -> bool:
        """检查日期范围内的数据是否都已下载"""
        dates = self.get_all_dates_in_range(range_start, range_end)
        return all(date in self.downloaded_dates for date in dates)
    
    def save_state(self, state: dict):
        """保存下载状态"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    def load_state(self) -> dict:
        """加载下载状态"""
        import json
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            logger.info(f"从断点恢复")
            return state
        return {
            'completed_months': [],
            'failed_months': [],
            'total_records': 0
        }
    
    def download_range(self, range_start: str, range_end: str, state: dict, desc: str = "日期范围"):
        """
        下载指定日期范围的数据
        
        Args:
            range_start: 起始日期 (YYYYMMDD)
            range_end: 结束日期 (YYYYMMDD)
            state: 状态字典
            desc: 描述（用于日志）
            
        Returns:
            (是否成功, 数据条数)
        """
        logger.info(f"  下载 {desc}: {range_start} - {range_end}")
        
        # 检查是否已下载
        if self.is_range_downloaded(range_start, range_end):
            logger.info(f"    ✓ 数据已存在，跳过")
            return True, 0
        
        try:
            limit = 5000
            offset = 0
            all_data = []
            
            while True:
                try:
                    df = self.ts_pro.daily_basic(
                        start_date=range_start,
                        end_date=range_end,
                        limit=limit,
                        offset=offset
                    )
                    
                    if df.empty:
                        logger.info(f"    ✓ 数据为空")
                        return True, 0
                    
                    logger.info(f"    ✓ 获取 {len(df)} 条记录")
                    all_data.append(df)
                    
                    # 更新offset
                    offset += len(df)
                    
                    # 如果获取的数据少于limit，说明已经是最后一页了
                    if len(df) < limit:
                        break
                        
                    # 检查是否超过单次调用限制
                    if offset >= self.max_records_per_call:
                        logger.warning(f"    ⚠️  数据量过大（{offset}条），无法完整下载")
                        return False, len(all_data)
                        
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"    ✗ 下载失败: {error_msg}")
                    # 如果是数据量相关的错误，返回False
                    if "查询数据失败" in error_msg or "数据量" in error_msg or "超过" in error_msg:
                        return False, len(all_data)
                    else:
                        # 其他错误，直接返回失败
                        return False, len(all_data)
            
            # 合并并保存数据
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                combined_df = combined_df.sort_values(['ts_code', 'trade_date'])
                
                # 追加到已有文件
                output_path = os.path.join(self.output_dir, "daily_basic_all.parquet")
                if os.path.exists(output_path):
                    existing_df = pd.read_parquet(output_path)
                    combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
                    combined_df = combined_df.sort_values(['ts_code', 'trade_date'])
                
                combined_df.to_parquet(output_path)
                logger.info(f"    ✓ 已保存: {len(combined_df)} 条记录")
                
                # 更新已下载日期
                self.downloaded_dates.update(combined_df['trade_date'].unique())
                
                return True, len(combined_df)
            else:
                return True, 0
                
        except Exception as e:
            logger.error(f"  ✗ 下载异常: {e}")
            return False, 0
    
    def download_month_with_retry(self, month_start: str, month_end: str, state: dict):
        """
        下载单个月份的数据（带重试逻辑）
        
        策略：
        1. 检查是否已下载
        2. 尝试按月下载
        3. 如果失败（数据量过大），按半月重试
        4. 失败不停止，继续下一个月份
        
        Args:
            month_start: 月份开始日期 (YYYYMMDD)
            month_end: 月份结束日期 (YYYYMMDD)
            state: 状态字典
            
        Returns:
            是否成功
        """
        logger.info(f"\n月份: {month_start} - {month_end}")
        
        # 检查是否已下载
        if self.is_range_downloaded(month_start, month_end):
            logger.info("  ✓ 数据已存在，跳过")
            state['completed_months'].append(month_start)
            self.save_state(state)
            return True
        
        # 检查是否已失败过
        if month_start in state['failed_months']:
            logger.info("  ✗ 之前失败过，跳过")
            return False
        
        # 策略1：尝试按月下载
        logger.info("  策略1: 按月下载")
        success, count = self.download_range(month_start, month_end, state, "月份")
        
        if success:
            logger.info(f"  ✓ 月份下载成功: {count} 条记录")
            state['completed_months'].append(month_start)
            state['total_records'] = state['total_records'] + count
            self.save_state(state)
            return True
        
        logger.warning(f"  ✗ 月份下载失败，尝试策略2")
        
        # 策略2：按半月重试
        logger.info("  策略2: 按半月下载")
        half_months = self.split_month_to_half_months(month_start, month_end)
        
        total_count = 0
        half_success = True
        
        for half_start, half_end in half_months:
            logger.info(f"  半月: {half_start} - {half_end}")
            success, count = self.download_range(half_start, half_end, state, "半月")
            
            if success:
                total_count += count
            else:
                logger.warning(f"  ✗ 半月下载失败: {half_start} - {half_end}")
                half_success = False
        
        if half_success:
            logger.info(f"  ✓ 半月下载成功: {total_count} 条记录")
            state['completed_months'].append(month_start)
            state['total_records'] = state['total_records'] + total_count
            self.save_state(state)
            return True
        
        # 所有策略都失败
        logger.error(f"  ✗ 所有策略都失败，标记为失败（但继续下载）")
        state['failed_months'].append(month_start)
        self.save_state(state)
        return False
    
    def download_all(self, resume: bool = True):
        """下载所有数据"""
        logger.info("=" * 70)
        logger.info("开始下载 daily_basic 数据（智能重试策略）")
        logger.info("=" * 70)
        logger.info(f"  日期范围: {self.start_date} - {self.end_date}")
        logger.info(f"  下载策略: 按月 → 按半月")
        logger.info(f"  已识别已下载数据: {len(self.downloaded_dates)} 个交易日")
        
        # 生成日期范围列表
        date_ranges = self.generate_month_ranges(self.start_date, self.end_date)
        logger.info(f"  共 {len(date_ranges)} 个月份需要检查")
        
        # 加载状态
        state = self.load_state() if resume else {
            'completed_months': [],
            'failed_months': [],
            'total_records': 0
        }
        
        logger.info(f"  状态文件已完成: {len(state['completed_months'])} 个月份")
        logger.info(f"  状态文件已失败: {len(state['failed_months'])} 个月份")
        logger.info(f"  状态文件记录数: {state['total_records']:,}")
        
        # 下载每个月份
        success_count = 0
        failed_count = 0
        skipped_count = 0
        
        for range_start, range_end in date_ranges:
            # 检查是否已下载（通过实际数据检查）
            if self.is_range_downloaded(range_start, range_end):
                skipped_count += 1
                if skipped_count % 10 == 0:
                    logger.info(f"  已跳过 {skipped_count} 个月份（数据已存在）")
                continue
            
            if range_start in state['completed_months']:
                skipped_count += 1
                continue
            
            if self.download_month_with_retry(range_start, range_end, state):
                success_count += 1
            else:
                failed_count += 1
            
            # 每下载5个月份显示进度
            if (success_count + failed_count) % 5 == 0:
                total = len(date_ranges)
                completed = len(state['completed_months'])
                logger.info(f"\n进度: {completed}/{total} ({completed/total*100:.1f}%)")
                logger.info(f"  成功: {success_count}, 失败: {failed_count}, 跳过: {skipped_count}")
        
        # 统计结果
        logger.info("\n" + "=" * 70)
        logger.info("下载完成！")
        logger.info("=" * 70)
        logger.info(f"  成功: {success_count}/{len(date_ranges)} 个月份")
        logger.info(f"  失败: {failed_count}/{len(date_ranges)} 个月份")
        logger.info(f"  跳过: {skipped_count}/{len(date_ranges)} 个月份（数据已存在）")
        logger.info(f"  总记录数: {state['total_records']:,}")
        
        if failed_count > 0:
            logger.info(f"\n失败的月份:")
            for month in state['failed_months']:
                logger.info(f"  - {month}")
        
        return state


def main():
    """主函数"""
    # 配置
    TUSHARE_TOKEN = get_tushare_token()
    START_DATE = "2020-01-01"
    END_DATE = "2026-02-09"
    
    # 创建下载器
    downloader = DailyBasicOptimizedDownloader(
        tushare_token=TUSHARE_TOKEN,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # 开始下载
    downloader.download_all(resume=True)


if __name__ == "__main__":
    main()
