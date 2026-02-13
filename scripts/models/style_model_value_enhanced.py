#!/usr/bin/env python3
"""
价值增强版风格模型
避免模型退化成"选高波动率小股票"的博弈模型
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


class ValueEnhancedStyleModel:
    """价值增强版风格模型"""

    def __init__(self, data_dir="data/tushare"):
        self.data_dir = Path(data_dir)
        self.load_data()

    def load_data(self):
        """加载数据"""
        logger.info("=" * 70)
        logger.info("加载数据...")
        logger.info("=" * 70)

        # 加载日线数据
        logger.info("加载日线数据...")
        daily_files = list(self.data_dir.glob("daily/daily_*.parquet"))
        self.daily = pd.concat([pd.read_parquet(f) for f in daily_files])
        self.daily['trade_date'] = pd.to_datetime(self.daily['trade_date'])
        logger.info(f"✓ 日线数据总计: {len(self.daily)} 条记录")

        # 加载daily_basic数据
        logger.info("加载daily_basic数据...")
        self.daily_basic = pd.read_parquet(self.data_dir / "daily_basic_all.parquet")
        self.daily_basic['trade_date'] = pd.to_datetime(self.daily_basic['trade_date'])
        logger.info(f"✓ Daily basic: {len(self.daily_basic)} 条记录")

        # 加载财务数据
        logger.info("加载财务数据...")
        fina_files = list(self.data_dir.glob("fundamental/fina_indicator_*.parquet"))
        self.fundamental = pd.concat([pd.read_parquet(f) for f in fina_files])
        self.fundamental['end_date'] = pd.to_datetime(self.fundamental['end_date'])
        logger.info(f"✓ 财务数据总计: {len(self.fundamental)} 条记录")

        # 合并数据
        logger.info("合并数据...")
        self.merged = self.daily.merge(
            self.daily_basic,
            on=['ts_code', 'trade_date'],
            how='left'
        )

        logger.info(f"✓ 合并后数据: {len(self.merged)} 条记录")

    def get_market_cap_threshold(self, market_state):
        """根据市场状态调整市值门槛"""
        if market_state == "HEALTHY_UP":
            return 50000000000   # 50亿
        elif market_state == "DISTRIBUTION":
            return 100000000000  # 100亿
        else:
            return 100000000000  # 默认100亿

    def check_fundamental_hard(self, stock_data):
        """基本面硬约束"""
        # ROE > 0
        indicator1 = stock_data['roe'] > 0

        # 净利润 > 0
        indicator2 = stock_data['net_profit'] > 0

        # 负债率 < 70%
        if 'debt_to_assets' in stock_data.columns:
            indicator3 = stock_data['debt_to_assets'] < 0.7
        else:
            indicator3 = pd.Series(True, index=stock_data.index)

        return indicator1 & indicator2 & indicator3

    def check_risk(self, stock_data):
        """风险约束"""
        # 过去60天最大回撤 < 40%
        stock_data['drawdown_60d'] = (
            (stock_data['close'] - stock_data['close'].rolling(60).max()) /
            stock_data['close'].rolling(60).max()
        )
        indicator1 = stock_data['drawdown_60d'] < 0.4

        # 波动率 < 30%
        volatility_60d = stock_data['pct_chg'].rolling(60).std()
        indicator2 = volatility_60d < 0.3

        return indicator1 & indicator2

    def calc_value_score(self, stock_data):
        """计算价值评分（0-100）"""
        # 1. 盈利能力（30分）
        if 'roe' in stock_data.columns:
            profit_score = min(stock_data['roe'] / 0.2, 1) * 30
        else:
            profit_score = 0

        # 2. 成长性（25分）
        if 'or_yoy' in stock_data.columns:  # 营收增长率
            growth_score = min(stock_data['or_yoy'] / 0.3, 1) * 25
        else:
            growth_score = 0

        # 3. 估值安全（25分）
        if 'pe_ttm' in stock_data.columns:
            valuation_score = min(1 / (stock_data['pe_ttm'] + 1e-8) / 50, 1) * 25
        else:
            valuation_score = 0

        # 4. 财务健康（20分）
        if 'debt_to_assets' in stock_data.columns:
            health_score = min((1 - stock_data['debt_to_assets']) / 0.7, 1) * 20
        else:
            health_score = 0

        total_score = profit_score + growth_score + valuation_score + health_score

        return total_score

    def select_stocks_value_enhanced(self, date_data, market_state="ACCUMULATION"):
        """
        价值增强版选股
        """
        # 第一步：硬性约束过滤
        cap_threshold = self.get_market_cap_threshold(market_state)
        filtered = date_data[
            (date_data['total_mv'] > cap_threshold) &
            (date_data['turnover_rate'] > 0.01)  # 换手率 > 1%
        ].copy()

        if len(filtered) == 0:
            return pd.DataFrame()

        # 第二步：基本面硬约束
        filtered = filtered[self.check_fundamental_hard(filtered)].copy()

        if len(filtered) == 0:
            return pd.DataFrame()

        # 第三步：风险约束
        filtered = filtered[self.check_risk(filtered)].copy()

        if len(filtered) == 0:
            return pd.DataFrame()

        # 第四步：计算动量信号
        filtered['ret_20d'] = filtered['close'].pct_change(20)
        filtered['amount_trend'] = (
            filtered['amount'].rolling(5).mean() /
            (filtered['amount'].rolling(20).mean() + 1e-8)
        )

        # 第五步：价值评分
        filtered['value_score'] = self.calc_value_score(filtered)

        # 第六步：综合评分
        filtered['final_score'] = (
            filtered['ret_20d'] * 100 +      # 20日收益率
            filtered['amount_trend'] * 20 +  # 成交额趋势
            filtered['value_score'] * 0.5    # 价值评分
        )

        # 第七步：筛选Top 30
        selected = filtered.nlargest(30, 'final_score')

        return selected

    def run(self):
        """运行模型"""
        logger.info("=" * 70)
        logger.info("开始价值增强版风格模型分析...")
        logger.info("=" * 70)

        # 获取所有交易日期
        all_dates = sorted(self.merged['trade_date'].unique())

        results = []

        for i, date in enumerate(all_dates):
            if i < 60:  # 需要至少60天历史数据
                continue

            if i % 20 == 0:
                logger.info(f"进度: {i}/{len(all_dates)}")

            date_data = self.merged[self.merged['trade_date'] == date].copy()

            if len(date_data) == 0:
                continue

            # 简化：假设市场状态为ACCUMULATION
            market_state = "ACCUMULATION"

            # 选股
            selected_stocks = self.select_stocks_value_enhanced(date_data, market_state)

            if len(selected_stocks) == 0:
                continue

            # 记录结果
            results.append({
                "trade_date": date,
                "selected_count": len(selected_stocks),
                "avg_mv": selected_stocks['total_mv'].mean(),
                "avg_roe": selected_stocks['roe'].mean() if 'roe' in selected_stocks.columns else 0,
                "avg_ret_20d": selected_stocks['ret_20d'].mean(),
                "avg_value_score": selected_stocks['value_score'].mean(),
                "top_stock": selected_stocks.iloc[0]['ts_code'] if len(selected_stocks) > 0 else None,
            })

        results_df = pd.DataFrame(results)

        # 保存结果
        output_file = self.data_dir / "factors" / "style_model_value_enhanced_results.parquet"
        results_df.to_parquet(output_file)
        logger.info(f"✓ 结果已保存: {output_file}")

        # 统计信息
        logger.info("=" * 70)
        logger.info("统计信息：")
        logger.info("=" * 70)
        logger.info(f"总交易日数: {len(results_df)}")
        logger.info(f"平均选股数: {results_df['selected_count'].mean():.2f}")
        logger.info(f"平均市值: {results_df['avg_mv'].mean() / 100000000:.2f} 亿")
        logger.info(f"平均ROE: {results_df['avg_roe'].mean():.2%}")
        logger.info(f"平均20日收益率: {results_df['avg_ret_20d'].mean():.2%}")
        logger.info(f"平均价值评分: {results_df['avg_value_score'].mean():.2f}")

        # 最新状态
        latest = results_df.iloc[-1]
        logger.info("=" * 70)
        logger.info("最新状态：")
        logger.info("=" * 70)
        logger.info(f"日期: {latest['trade_date'].strftime('%Y-%m-%d')}")
        logger.info(f"选股数: {latest['selected_count']}")
        logger.info(f"平均市值: {latest['avg_mv'] / 100000000:.2f} 亿")
        logger.info(f"平均ROE: {latest['avg_roe']:.2%}")
        logger.info(f"平均20日收益率: {latest['avg_ret_20d']:.2%}")
        logger.info(f"平均价值评分: {latest['avg_value_score']:.2f}")

        return results_df


def main():
    """主函数"""
    model = ValueEnhancedStyleModel()
    results = model.run()

    logger.info("=" * 70)
    logger.info("✓ 价值增强版风格模型分析完成！")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()