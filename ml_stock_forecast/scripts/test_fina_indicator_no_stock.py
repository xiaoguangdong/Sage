#!/usr/bin/env python3
"""
测试财务指标接口是否支持批量调用（不传股票代码）
"""

import sys
import os
import time
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ml_stock_forecast.data.data_provider import DataProvider

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_fina_indicator_no_stock():
    """测试fina_indicator接口不传股票代码"""
    token = "2bcc0e9feb650d9862330a9743e5cc2e6469433c4d1ea0ce2d79371e"
    provider = DataProvider(tushare_token=token)

    logger.info("=" * 70)
    logger.info("测试fina_indicator接口 - 不传股票代码")
    logger.info("=" * 70)

    logger.info("等待10秒...")
    time.sleep(10)

    try:
        # 测试1：不传股票代码，只传日期
        logger.info("\n测试1: 不传股票代码，只传日期范围")
        df = provider.ts_pro.fina_indicator(
            start_date='20200101',
            end_date='20200131'
        )
        
        if df.empty:
            logger.info("  返回数据为空")
        else:
            logger.info(f"  ✓ 成功获取 {len(df)} 条记录")
            logger.info(f"  列名: {df.columns.tolist()}")
            if 'ts_code' in df.columns:
                logger.info(f"  包含股票代码，返回的是单只股票数据")
                logger.info(f"  股票代码: {df['ts_code'].unique()}")
            else:
                logger.info(f"  不包含股票代码，返回的是汇总数据")

    except Exception as e:
        logger.error(f"  ✗ 失败: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("测试完成")
    logger.info("=" * 70)

if __name__ == "__main__":
    test_fina_indicator_no_stock()