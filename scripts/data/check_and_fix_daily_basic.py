#!/usr/bin/env python3
"""
检查和修复daily_basic数据

功能：
1. 读取parquet文件并去重
2. 按日期分组统计每天的股票数量
3. 识别缺失的日期
4. 重新下载缺失的数据
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


class DailyBasicCheckerAndFixer:
    """daily_basic数据检查和修复工具"""
    
    def __init__(self, tushare_token: str, start_date: str = "2020-01-01", end_date: str = "2026-02-09"):
        """
        初始化检查器
        
        Args:
            tushare_token: Tushare token
            start_date: 开始日期
            end_date: 结束日期
        """
        self.tushare_token = tushare_token
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = "data/tushare"
        
        # 初始化Tushare Pro
        self.ts_pro = ts.pro_api(tushare_token)
        logger.info("Tushare Pro初始化成功")
        
        # 数据文件路径
        self.data_file = os.path.join(self.output_dir, "daily_basic_all.parquet")
        self.backup_file = os.path.join(self.output_dir, "daily_basic_all_backup.parquet")
    
    def check_data(self):
        """检查数据完整性"""
        logger.info("=" * 70)
        logger.info("开始检查 daily_basic 数据完整性")
        logger.info("=" * 70)
        
        # 1. 读取数据
        if not os.path.exists(self.data_file):
            logger.error(f"数据文件不存在: {self.data_file}")
            return None
        
        logger.info("读取数据文件...")
        df = pd.read_parquet(self.data_file)
        logger.info(f"原始记录数: {len(df):,}")
        
        # 2. 去重
        logger.info("去重处理...")
        df_dedup = df.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
        logger.info(f"去重后记录数: {len(df_dedup):,}")
        logger.info(f"删除重复记录: {len(df) - len(df_dedup):,}")
        
        # 3. 按日期统计
        logger.info("\n按日期统计股票数量:")
        date_stats = df_dedup.groupby('trade_date').size().sort_index()
        
        # 显示统计信息
        print("\n按年份统计:")
        print(df_dedup.groupby(df_dedup['trade_date'].str[:4]).size())
        
        print("\n按月份统计（最近10个月）:")
        print(date_stats.tail(10))
        
        # 4. 识别缺失日期
        logger.info("\n检查缺失日期...")
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # 生成所有可能的交易日期
        all_dates = pd.date_range(start=start, end=end, freq='B')  # B表示工作日
        
        # 转换为字符串格式
        all_dates_str = [d.strftime('%Y%m%d') for d in all_dates]
        
        # 获取已有的日期
        existing_dates = set(df_dedup['trade_date'].unique())
        
        # 找出缺失的日期
        missing_dates = sorted(set(all_dates_str) - existing_dates)
        
        logger.info(f"理论交易天数: {len(all_dates_str)}")
        logger.info(f"实际交易天数: {len(existing_dates)}")
        logger.info(f"缺失天数: {len(missing_dates)}")
        
        if missing_dates:
            logger.info(f"\n缺失日期列表（前20个）:")
            for date in missing_dates[:20]:
                logger.info(f"  - {date}")
            if len(missing_dates) > 20:
                logger.info(f"  ... 还有 {len(missing_dates) - 20} 个缺失日期")
        else:
            logger.info("✓ 没有缺失日期")
        
        # 5. 识别数据量异常的日期（股票数少于5000）
        logger.info("\n检查数据量异常的日期（股票数 < 5000）:")
        low_count_dates = date_stats[date_stats < 5000]
        
        if not low_count_dates.empty:
            logger.info(f"发现 {len(low_count_dates)} 个数据量异常的日期:")
            for date, count in low_count_dates.items():
                logger.info(f"  - {date}: {count} 只股票")
        else:
            logger.info("✓ 没有数据量异常的日期")
        
        # 6. 生成需要重新下载的日期列表
        need_download = sorted(missing_dates + list(low_count_dates.index))
        
        logger.info(f"\n总共需要重新下载的日期: {len(need_download)} 天")
        
        # 保存检查结果
        result = {
            'total_records': len(df_dedup),
            'date_range': f"{df_dedup['trade_date'].min()} - {df_dedup['trade_date'].max()}",
            'stock_count': df_dedup['ts_code'].nunique(),
            'trading_days': len(existing_dates),
            'missing_dates': missing_dates,
            'low_count_dates': low_count_dates.to_dict(),
            'need_download': need_download
        }
        
        # 保存到文件
        import json
        result_file = os.path.join(self.output_dir, "check_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"\n检查结果已保存到: {result_file}")
        
        return result
    
    def backup_data(self):
        """备份数据"""
        logger.info("\n备份数据文件...")
        if os.path.exists(self.data_file):
            import shutil
            shutil.copy2(self.data_file, self.backup_file)
            logger.info(f"数据已备份到: {self.backup_file}")
        else:
            logger.warning("数据文件不存在，无需备份")
    
    def fix_data(self, result: dict):
        """
        修复缺失的数据
        
        Args:
            result: 检查结果字典
        """
        if not result or not result['need_download']:
            logger.info("没有需要修复的数据")
            return
        
        need_download = result['need_download']
        logger.info(f"\n开始修复 {len(need_download)} 天的数据")
        
        # 备份现有数据
        self.backup_data()
        
        # 读取现有数据
        df_existing = pd.read_parquet(self.data_file)
        logger.info(f"现有数据: {len(df_existing):,} 条")
        
        # 下载缺失的数据
        df_new = []
        success_count = 0
        failed_dates = []
        
        for i, date in enumerate(need_download):
            logger.info(f"\n[{i+1}/{len(need_download)}] 下载日期: {date}")
            
            try:
                df = self.ts_pro.daily_basic(
                    trade_date=date,
                    limit=10000,
                    offset=0
                )
                
                if df is not None and not df.empty:
                    logger.info(f"  ✓ 下载成功: {len(df)} 条记录")
                    df_new.append(df)
                    success_count += 1
                else:
                    logger.warning(f"  ⚠️  该日期无数据")
                    failed_dates.append(date)
                
                # 避免请求过快
                import time
                time.sleep(0.3)
                
            except Exception as e:
                logger.error(f"  ✗ 下载失败: {e}")
                failed_dates.append(date)
        
        # 合并数据
        if df_new:
            df_combined = pd.concat(df_new, ignore_index=True)
            logger.info(f"\n新下载的数据: {len(df_combined):,} 条")
            
            # 合并现有数据和新数据
            df_all = pd.concat([df_existing, df_combined], ignore_index=True)
            
            # 去重
            df_all = df_all.drop_duplicates(subset=['ts_code', 'trade_date'], keep='first')
            
            # 排序
            df_all = df_all.sort_values(['trade_date', 'ts_code'])
            
            # 保存
            df_all.to_parquet(self.data_file)
            logger.info(f"✓ 数据已保存: {len(df_all):,} 条记录")
        
        # 保存修复结果
        fix_result = {
            'total_dates': len(need_download),
            'success_dates': success_count,
            'failed_dates': failed_dates,
            'new_records': len(df_combined) if df_new else 0,
            'final_records': len(df_all) if df_new else len(df_existing)
        }
        
        import json
        fix_result_file = os.path.join(self.output_dir, "fix_result.json")
        with open(fix_result_file, 'w', encoding='utf-8') as f:
            json.dump(fix_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n修复完成！")
        logger.info(f"  成功: {success_count} 天")
        logger.info(f"  失败: {len(failed_dates)} 天")
        logger.info(f"  新增记录: {fix_result['new_records']:,} 条")
        logger.info(f"  最终记录: {fix_result['final_records']:,} 条")


def main():
    """主函数"""
    # 配置
    TUSHARE_TOKEN = get_tushare_token()
    START_DATE = "2020-01-01"
    END_DATE = "2026-02-09"
    
    # 创建检查器
    checker = DailyBasicCheckerAndFixer(
        tushare_token=TUSHARE_TOKEN,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # 检查数据
    result = checker.check_data()
    
    # 询问是否修复
    if result and result['need_download']:
        print("\n" + "=" * 70)
        print("检查完成！发现缺失的数据")
        print("=" * 70)
        print(f"需要重新下载的日期: {len(result['need_download'])} 天")
        
        # 自动修复
        checker.fix_data(result)
    else:
        print("\n" + "=" * 70)
        print("检查完成！数据完整")
        print("=" * 70)


if __name__ == "__main__":
    main()
