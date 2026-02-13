#!/usr/bin/env python3
"""
自适应因子权重系统

根据市场风格状态动态调整因子权重
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class AdaptiveFactorWeightSystem:
    """自适应因子权重系统"""
    
    def __init__(self):
        self.data_dir = "data/tushare"
        self.output_dir = "data/tushare/factors"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("自适应因子权重系统初始化")
    
    def load_data(self):
        """加载所有必要的数据"""
        logger.info("=" * 70)
        logger.info("加载数据...")
        logger.info("=" * 70)
        
        # 加载因子数据
        factors_file = os.path.join(self.output_dir, "stock_factors_with_score.parquet")
        if os.path.exists(factors_file):
            self.factors = pd.read_parquet(factors_file)
            self.factors['trade_date'] = pd.to_datetime(self.factors['trade_date'])
            logger.info(f"✓ 因子数据: {len(self.factors)} 条记录")
        else:
            logger.error("✗ 未找到因子数据")
            return False
        
        # 加载风格特征数据
        regime_file = os.path.join(self.output_dir, "market_regime_features.parquet")
        if os.path.exists(regime_file):
            self.regime = pd.read_parquet(regime_file)
            self.regime['trade_date'] = pd.to_datetime(self.regime['trade_date'])
            logger.info(f"✓ 风格特征: {len(self.regime)} 条记录")
        else:
            logger.error("✗ 未找到风格特征数据")
            return False
        
        return True
    
    def calculate_adaptive_weights(self):
        """计算自适应权重"""
        logger.info("\n" + "=" * 70)
        logger.info("计算自适应权重...")
        logger.info("=" * 70)
        
        # 为每个日期计算因子权重
        all_dates = sorted(self.factors['trade_date'].unique())
        
        weight_history = []
        
        for trade_date in all_dates:
            # 获取该日期的风格状态
            regime_data = self.regime[self.regime['trade_date'] == trade_date]
            
            if len(regime_data) == 0:
                # 如果没有风格数据，使用默认权重
                trend = 0
                liquidity = 0
                fundamental = 0
                speculative = 0
            else:
                trend = regime_data['trend_raw'].values[0]
                liquidity = regime_data['liquidity_raw'].values[0]
                fundamental = regime_data['fundamental_raw'].values[0]
                speculative = regime_data['speculative_raw'].values[0]
            
            # 根据风格状态调整因子权重
            # 标准化风格分数到 0-1 范围
            trend_score = 1 / (1 + np.exp(-trend / 2))  # sigmoid
            liquidity_score = 1 / (1 + np.exp(-liquidity / 2))
            fundamental_score = 1 / (1 + np.exp(-fundamental / 0.1))
            speculative_score = 1 / (1 + np.exp(-speculative / 0.5))
            
            # 动量因子权重（4w_ret, 12w_ret）
            # 趋势市（trend_raw > 0）增加权重，反转市减少权重
            momentum_weight = 0.3 + 0.5 * trend_score
            
            # 质量因子权重（roe_ttm, gross_margin）
            # 基本面市增加权重
            quality_weight = 0.2 + 0.3 * fundamental_score
            
            # 流动性因子权重（turnover, amt_rank）
            # 流动性驱动时增加权重
            liquidity_factor_weight = 0.2 + 0.3 * liquidity_score
            
            # 风险因子权重（vol_12w, beta）
            # 投机市时降低风险因子的权重（因为要冒险）
            risk_weight = 0.3 - 0.2 * speculative_score
            
            # 归一化权重
            total_weight = momentum_weight + quality_weight + liquidity_factor_weight + risk_weight
            momentum_weight /= total_weight
            quality_weight /= total_weight
            liquidity_factor_weight /= total_weight
            risk_weight /= total_weight
            
            weight_history.append({
                'trade_date': trade_date,
                'trend_raw': trend,
                'liquidity_raw': liquidity,
                'fundamental_raw': fundamental,
                'speculative_raw': speculative,
                'trend_score': trend_score,
                'liquidity_score': liquidity_score,
                'fundamental_score': fundamental_score,
                'speculative_score': speculative_score,
                'momentum_weight': momentum_weight,
                'quality_weight': quality_weight,
                'liquidity_factor_weight': liquidity_factor_weight,
                'risk_weight': risk_weight,
            })
        
        self.weights_df = pd.DataFrame(weight_history)
        logger.info(f"✓ 权重计算完成: {len(self.weights_df)} 条记录")
        
        return self.weights_df
    
    def calculate_adaptive_scores(self):
        """计算自适应调整后的得分"""
        logger.info("\n" + "=" * 70)
        logger.info("计算自适应得分...")
        logger.info("=" * 70)
        
        # 合并因子数据和权重数据
        merged = self.factors.merge(self.weights_df, on='trade_date', how='left')
        
        # 计算每个因子的排名得分（0-100）
        results = []
        
        for trade_date, group in merged.groupby('trade_date'):
            # 排名规则：值越大越好
            group['rank_4w_ret'] = group['4w_ret'].rank(pct=True) * 100
            group['rank_12w_ret'] = group['12w_ret'].rank(pct=True) * 100
            group['rank_roe_ttm'] = group['roe_ttm'].rank(pct=True) * 100
            group['rank_gross_margin'] = group['gross_margin'].rank(pct=True) * 100
            group['rank_turnover'] = 100 - abs(group['turnover'] - group['turnover'].median()).rank(pct=True) * 100
            group['rank_amt_rank'] = 100 - group['amt_rank'].rank(pct=True) * 100
            group['rank_vol_12w'] = 100 - group['vol_12w'].rank(pct=True) * 100
            group['rank_beta'] = 100 - abs(group['beta'] - 1).rank(pct=True) * 100
            
            # 使用自适应权重计算综合得分
            momentum_score = (group['rank_4w_ret'] + group['rank_12w_ret']) / 2
            quality_score = (group['rank_roe_ttm'] + group['rank_gross_margin']) / 2
            liquidity_score = (group['rank_turnover'] + group['rank_amt_rank']) / 2
            risk_score = (group['rank_vol_12w'] + group['rank_beta']) / 2
            
            # 加权综合得分
            group['adaptive_score'] = (
                momentum_score * group['momentum_weight'] +
                quality_score * group['quality_weight'] +
                liquidity_score * group['liquidity_factor_weight'] +
                risk_score * group['risk_weight']
            )
            
            results.append(group)
        
        self.adaptive_scores = pd.concat(results, ignore_index=True)
        logger.info(f"✓ 自适应得分计算完成: {len(self.adaptive_scores)} 条记录")
        
        return self.adaptive_scores
    
    def save_results(self):
        """保存结果"""
        logger.info("\n" + "=" * 70)
        logger.info("保存结果...")
        logger.info("=" * 70)
        
        # 保存权重历史
        weight_file = os.path.join(self.output_dir, "adaptive_factor_weights.csv")
        self.weights_df.to_csv(weight_file, index=False)
        logger.info(f"✓ 权重历史: {weight_file}")
        
        # 保存自适应得分
        score_file = os.path.join(self.output_dir, "stock_factors_with_adaptive_score.parquet")
        self.adaptive_scores.to_parquet(score_file)
        logger.info(f"✓ 自适应得分: {score_file}")
        
        # 保存最新得分
        latest_date = self.adaptive_scores['trade_date'].max()
        latest_df = self.adaptive_scores[self.adaptive_scores['trade_date'] == latest_date].copy()
        latest_df = latest_df.sort_values('adaptive_score', ascending=False)
        
        latest_file = os.path.join(self.output_dir, f"latest_adaptive_scores_{latest_date.strftime('%Y%m%d')}.csv")
        latest_df.to_csv(latest_file, index=False)
        logger.info(f"✓ 最新自适应得分: {latest_file}")
        
        # 输出统计摘要
        logger.info("\n" + "=" * 70)
        logger.info("权重统计摘要：")
        logger.info("=" * 70)
        
        weight_cols = ['momentum_weight', 'quality_weight', 'liquidity_factor_weight', 'risk_weight']
        for col in weight_cols:
            valid_data = self.weights_df[col].dropna()
            logger.info(f"\n{col}:")
            logger.info(f"  平均值: {valid_data.mean():.4f}")
            logger.info(f"  标准差: {valid_data.std():.4f}")
            logger.info(f"  最小值: {valid_data.min():.4f}")
            logger.info(f"  最大值: {valid_data.max():.4f}")
        
        # 显示最新的权重
        logger.info("\n" + "=" * 70)
        logger.info("最新权重状态（最新10个周期）：")
        logger.info("=" * 70)
        latest_weights = self.weights_df.tail(10)[['trade_date', 'trend_raw', 'momentum_weight', 'quality_weight', 'liquidity_factor_weight', 'risk_weight']]
        print(latest_weights.to_string(index=False))
        
        # 显示最新Top 20股票
        logger.info("\n" + "=" * 70)
        logger.info("最新自适应得分Top 20股票：")
        logger.info("=" * 70)
        print(latest_df[['ts_code', 'adaptive_score', 'momentum_weight', '4w_ret', '12w_ret', 'roe_ttm']].head(20).to_string(index=False))
    
    def run(self):
        """执行完整的自适应权重调整流程"""
        logger.info("\n" + "=" * 70)
        logger.info("开始自适应权重调整...")
        logger.info("=" * 70)
        
        if not self.load_data():
            logger.error("数据加载失败")
            return None
        
        if not self.calculate_adaptive_weights():
            logger.error("权重计算失败")
            return None
        
        if not self.calculate_adaptive_scores():
            logger.error("自适应得分计算失败")
            return None
        
        self.save_results()
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ 自适应权重调整完成！")
        logger.info("=" * 70)
        
        return self.adaptive_scores


def main():
    """主函数"""
    system = AdaptiveFactorWeightSystem()
    adaptive_scores = system.run()
    
    if adaptive_scores is not None:
        print("\n" + "=" * 70)
        print("使用说明：")
        print("=" * 70)
        print("1. trend_raw > 0: 趋势市，momentum_weight 自动增加")
        print("2. trend_raw < 0: 反转市，momentum_weight 自动降低")
        print("3. fundamental_raw > 0: 基本面市，quality_weight 自动增加")
        print("4. speculative_raw > 0: 投机市，risk_weight 自动降低（鼓励冒险）")
        print("\n自适应得分相比固定得分，能更好地适应市场风格切换！")


if __name__ == "__main__":
    main()