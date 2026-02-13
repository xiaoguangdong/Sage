#!/usr/bin/env python3
"""
价值增强版回测：测试胜率
基于之前的因子数据，加入价值约束，测试胜率是否提升
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backtest_with_value_filter():
    """回测：对比有无价值约束的胜率"""
    logger.info("=" * 70)
    logger.info("开始价值增强版回测...")
    logger.info("=" * 70)

    # 读取因子数据
    logger.info("加载因子数据...")
    factors = pd.read_parquet('data/tushare/factors/stock_factors_with_score.parquet')
    factors['trade_date'] = pd.to_datetime(factors['trade_date'])
    logger.info(f"✓ 因子数据: {len(factors)} 条记录")

    # 读取日线数据用于计算未来收益
    logger.info("加载日线数据...")
    daily_files = list(Path('data/tushare/daily').glob('daily_*.parquet'))
    daily = pd.concat([pd.read_parquet(f) for f in daily_files])
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    logger.info(f"✓ 日线数据: {len(daily)} 条记录")

    # 选择测试日期（按月）
    all_dates = sorted(factors['trade_date'].unique())
    test_dates = [d for i, d in enumerate(all_dates) if i >= 60 and i % 20 == 0]
    latest_date = daily['trade_date'].max()
    valid_dates = [d for d in test_dates if (latest_date - d).days >= 70]

    logger.info(f"选择 {len(valid_dates)} 个测试日期")

    results = []

    for i, test_date in enumerate(valid_dates):
        if (i + 1) % 10 == 0:
            logger.info(f"进度: {i+1}/{len(valid_dates)}")

        date_factors = factors[factors['trade_date'] == test_date].copy()

        if len(date_factors) == 0:
            continue

        # ========== 无价值约束 ==========
        # 选择Top 30高分股票
        top30_no_filter = date_factors.nlargest(30, 'score')['ts_code'].tolist()

        # ========== 有价值约束 ==========
        # 1. 市值 > 50亿（500000万元）
        date_factors_mv = date_factors[date_factors['total_mv'] > 500000].copy()

        # 2. PE > 0 且 < 50
        date_factors_pe = date_factors_mv[
            (date_factors_mv['pe_ttm'] > 0) & (date_factors_mv['pe_ttm'] < 50)
        ].copy()

        # 3. PB > 0 且 < 10
        date_factors_pb = date_factors_pe[
            (date_factors_pe['pb'] > 0) & (date_factors_pe['pb'] < 10)
        ].copy()

        if len(date_factors_pb) > 0:
            top30_with_filter = date_factors_pb.nlargest(30, 'score')['ts_code'].tolist()
        else:
            top30_with_filter = []

        # ========== 计算未来收益 ==========
        def calc_future_return(stocks, days):
            returns = []
            for stock in stocks:
                stock_data = daily[
                    (daily['ts_code'] == stock) & (daily['trade_date'] > test_date)
                ].sort_values('trade_date')

                if len(stock_data) > days:
                    start_price = stock_data.iloc[0]['close']
                    end_price = stock_data.iloc[days]['close']
                    ret = (end_price - start_price) / start_price * 100
                    returns.append(ret)

            return np.mean(returns) if len(returns) > 0 else np.nan

        # 计算1周、4周、12周收益
        ret_1w_no_filter = calc_future_return(top30_no_filter, 5)
        ret_4w_no_filter = calc_future_return(top30_no_filter, 20)
        ret_12w_no_filter = calc_future_return(top30_no_filter, 60)

        if len(top30_with_filter) > 0:
            ret_1w_with_filter = calc_future_return(top30_with_filter, 5)
            ret_4w_with_filter = calc_future_return(top30_with_filter, 20)
            ret_12w_with_filter = calc_future_return(top30_with_filter, 60)
        else:
            ret_1w_with_filter = np.nan
            ret_4w_with_filter = np.nan
            ret_12w_with_filter = np.nan

        results.append({
            "trade_date": test_date,
            "no_filter_count": len(top30_no_filter),
            "with_filter_count": len(top30_with_filter),
            "ret_1w_no_filter": ret_1w_no_filter,
            "ret_4w_no_filter": ret_4w_no_filter,
            "ret_12w_no_filter": ret_12w_no_filter,
            "ret_1w_with_filter": ret_1w_with_filter,
            "ret_4w_with_filter": ret_4w_with_filter,
            "ret_12w_with_filter": ret_12w_with_filter,
        })

    results_df = pd.DataFrame(results)

    # 统计对比
    logger.info("=" * 70)
    logger.info("胜率对比：")
    logger.info("=" * 70)

    # 无价值约束
    no_filter_win_1w = (results_df['ret_1w_no_filter'] > 0).sum() / results_df['ret_1w_no_filter'].notna().sum()
    no_filter_win_4w = (results_df['ret_4w_no_filter'] > 0).sum() / results_df['ret_4w_no_filter'].notna().sum()
    no_filter_win_12w = (results_df['ret_12w_no_filter'] > 0).sum() / results_df['ret_12w_no_filter'].notna().sum()

    # 有价值约束
    with_filter_win_1w = (results_df['ret_1w_with_filter'] > 0).sum() / results_df['ret_1w_with_filter'].notna().sum()
    with_filter_win_4w = (results_df['ret_4w_with_filter'] > 0).sum() / results_df['ret_4w_with_filter'].notna().sum()
    with_filter_win_12w = (results_df['ret_12w_with_filter'] > 0).sum() / results_df['ret_12w_with_filter'].notna().sum()

    logger.info("\n无价值约束：")
    logger.info(f"  1周胜率: {no_filter_win_1w:.2%}")
    logger.info(f"  4周胜率: {no_filter_win_4w:.2%}")
    logger.info(f"  12周胜率: {no_filter_win_12w:.2%}")

    logger.info("\n有价值约束：")
    logger.info(f"  1周胜率: {with_filter_win_1w:.2%}")
    logger.info(f"  4周胜率: {with_filter_win_4w:.2%}")
    logger.info(f"  12周胜率: {with_filter_win_12w:.2%}")

    # 平均收益对比
    logger.info("\n平均收益对比：")
    logger.info(f"无价值约束 - 1周: {results_df['ret_1w_no_filter'].mean():.2%}")
    logger.info(f"无价值约束 - 4周: {results_df['ret_4w_no_filter'].mean():.2%}")
    logger.info(f"无价值约束 - 12周: {results_df['ret_12w_no_filter'].mean():.2%}")

    logger.info(f"\n有价值约束 - 1周: {results_df['ret_1w_with_filter'].mean():.2%}")
    logger.info(f"有价值约束 - 4周: {results_df['ret_4w_with_filter'].mean():.2%}")
    logger.info(f"有价值约束 - 12周: {results_df['ret_12w_with_filter'].mean():.2%}")

    # 胜率提升
    logger.info("\n胜率提升：")
    logger.info(f"  1周: {with_filter_win_1w - no_filter_win_1w:+.2%}")
    logger.info(f"  4周: {with_filter_win_4w - no_filter_win_4w:+.2%}")
    logger.info(f"  12周: {with_filter_win_12w - no_filter_win_12w:+.2%}")

    # 保存结果
    output_file = 'data/tushare/factors/value_enhanced_backtest_results.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ 结果已保存: {output_file}")

    logger.info("=" * 70)
    logger.info("✓ 回测完成！")
    logger.info("=" * 70)


if __name__ == "__main__":
    backtest_with_value_filter()