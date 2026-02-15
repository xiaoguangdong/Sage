"""
风险控制模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class RiskControl:
    """风险控制类"""
    
    def __init__(self, config: dict = None):
        """
        初始化风险控制
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 默认配置
        self.default_config = {
            'stop_loss': -0.08,  # 止损比例（-8%）
            'max_drawdown': -0.15,  # 最大回撤（-15%）
            'max_single_position': 0.05,  # 单只股票最大仓位（5%）
            'max_sector_exposure': 0.3,  # 单个行业最大暴露（30%）
            'volatility_threshold': 0.05,  # 波动率阈值
            'enable_stop_loss': True,  # 启用止损
            'enable_drawdown_control': True,  # 启用回撤控制
            'enable_position_limit': True,  # 启用仓位限制
            'preserve_gross_exposure': True,  # 调权后保持总仓位
            # 豆包风格仓位管理（简化门控）
            'target_volatility': {
                2: 0.18,  # RISK_ON
                1: 0.12,  # 震荡
                0: 0.08,  # RISK_OFF
            },
            'position_caps': {
                2: 1.00,
                1: 0.60,
                0: 0.30,
            },
            'market_vol_window': 20,
            'daily_shock_threshold': -0.02,
            'weekly_shock_threshold': -0.05,
            'daily_shock_cut': 0.10,
            'weekly_shock_cut': 0.50,
            'drawdown_cuts': [
                {'threshold': -0.15, 'cut': 0.90},
                {'threshold': -0.12, 'cut': 0.60},
                {'threshold': -0.08, 'cut': 0.30},
            ],
        }
        
        # 合并配置
        self.config = {**self.default_config, **self.config}
        
        # 运行时状态
        self.current_drawdown = 0.0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
    
    def compute_market_volatility(self, index_df: pd.DataFrame) -> float:
        """
        计算市场年化波动率（基于近N日）
        """
        if index_df is None or index_df.empty:
            return np.nan
        window = int(self.config.get('market_vol_window', 20))
        temp = index_df.copy().sort_values('trade_date')

        if 'pct_chg' in temp.columns:
            returns = pd.to_numeric(temp['pct_chg'], errors='coerce') / 100.0
        elif 'return' in temp.columns:
            returns = pd.to_numeric(temp['return'], errors='coerce')
        elif 'close' in temp.columns:
            close = pd.to_numeric(temp['close'], errors='coerce')
            returns = close.pct_change()
        else:
            return np.nan

        returns = returns.dropna()
        if len(returns) < max(5, window):
            return np.nan

        annualized_vol = returns.tail(window).std(ddof=0) * np.sqrt(252)
        return float(annualized_vol)
    
    def compute_target_position(
        self,
        trend_state: int,
        market_volatility: float,
        latest_index_return: float | None = None,
        latest_week_return: float | None = None,
        portfolio_drawdown: float | None = None,
    ) -> Dict[str, float]:
        """
        基于趋势状态 + 波动率 + 冲击事件计算目标仓位
        """
        target_volatility_cfg = self.config.get('target_volatility', {})
        position_caps_cfg = self.config.get('position_caps', {})
        target_vol = float(target_volatility_cfg.get(trend_state, target_volatility_cfg.get(1, 0.12)))
        position_cap = float(position_caps_cfg.get(trend_state, position_caps_cfg.get(1, 0.60)))

        if market_volatility is None or np.isnan(market_volatility) or market_volatility <= 0:
            base_position = position_cap
        else:
            base_position = min(target_vol / market_volatility, position_cap)

        position_after_shock = float(base_position)
        daily_threshold = float(self.config.get('daily_shock_threshold', -0.02))
        weekly_threshold = float(self.config.get('weekly_shock_threshold', -0.05))
        daily_cut = float(self.config.get('daily_shock_cut', 0.10))
        weekly_cut = float(self.config.get('weekly_shock_cut', 0.50))

        if latest_index_return is not None and latest_index_return <= daily_threshold:
            position_after_shock *= max(0.0, 1.0 - daily_cut)
        if latest_week_return is not None and latest_week_return <= weekly_threshold:
            position_after_shock *= max(0.0, 1.0 - weekly_cut)

        position_after_drawdown = float(position_after_shock)
        drawdown_rules = sorted(
            self.config.get('drawdown_cuts', []),
            key=lambda item: item.get('threshold', 0),
        )
        if portfolio_drawdown is not None:
            for rule in drawdown_rules:
                threshold = float(rule.get('threshold', -1.0))
                cut = float(rule.get('cut', 0.0))
                if portfolio_drawdown <= threshold:
                    position_after_drawdown *= max(0.0, 1.0 - cut)
                    break

        final_position = float(np.clip(position_after_drawdown, 0.0, 1.0))
        return {
            'trend_state': int(trend_state),
            'market_volatility': float(market_volatility) if market_volatility is not None and not np.isnan(market_volatility) else np.nan,
            'base_position': float(base_position),
            'position_after_shock': float(position_after_shock),
            'final_position': final_position,
        }
    
    def scale_to_target_exposure(self, df_portfolio: pd.DataFrame, target_exposure: float) -> pd.DataFrame:
        """
        将组合总仓位缩放到目标仓位
        """
        result = df_portfolio.copy()
        target_exposure = float(np.clip(target_exposure, 0.0, 1.0))
        if result.empty or 'weight' not in result.columns:
            return result
        current_exposure = float(result['weight'].sum())
        if current_exposure <= 0:
            return result
        result['weight'] = result['weight'] * (target_exposure / current_exposure)
        return result
    
    def check_entry_signal(self, df: pd.DataFrame, entry_signal: pd.Series) -> pd.Series:
        """
        检查买入信号是否应该被过滤
        
        Args:
            df: 股票数据
            entry_signal: 原始买入信号
            
        Returns:
            过滤后的买入信号
        """
        filtered_signal = entry_signal.copy()
        
        # 检查波动率
        if 'volatility_5d' in df.columns:
            high_vol = df['volatility_5d'] > self.config['volatility_threshold']
            filtered_signal = filtered_signal & (~high_vol)
            logger.info(f"波动率过滤: 过滤掉{high_vol.sum()}个信号")
        
        # 检查价格位置
        if 'price_ma20_ratio' in df.columns:
            below_ma = df['price_ma20_ratio'] < 0.95
            filtered_signal = filtered_signal & (~below_ma)
            logger.info(f"价格位置过滤: 过滤掉{below_ma.sum()}个信号")
        
        return filtered_signal
    
    def check_exit_signal(self, df: pd.DataFrame, current_positions: pd.Series) -> pd.Series:
        """
        检查是否需要止损或止盈
        
        Args:
            df: 股票数据
            current_positions: 当前持仓数据，包含买入价格
            
        Returns:
            卖出信号
        """
        exit_signal = pd.Series(0, index=df.index)
        
        if self.config['enable_stop_loss']:
            # 计算当前收益率
            if 'entry_price' in df.columns:
                current_return = (df['close'] - df['entry_price']) / df['entry_price']
                
                # 止损检查
                stop_loss = current_return < self.config['stop_loss']
                exit_signal = exit_signal | stop_loss
                
                if stop_loss.sum() > 0:
                    logger.info(f"止损触发: {stop_loss.sum()}只股票")
        
        return exit_signal
    
    def check_portfolio_risk(self, df_portfolio: pd.DataFrame) -> Dict[str, bool]:
        """
        检查组合风险
        
        Args:
            df_portfolio: 组合数据，包含'weight'列
            
        Returns:
            风险检查结果字典
        """
        risk_checks = {}
        
        # 检查单只股票仓位
        if self.config['enable_position_limit']:
            max_position = df_portfolio['weight'].max()
            risk_checks['single_position_ok'] = max_position <= self.config['max_single_position']
            
            if not risk_checks['single_position_ok']:
                logger.warning(f"单只股票仓位{max_position:.2%}超过限制{self.config['max_single_position']:.2%}")
        
        # 检查行业暴露
        if 'sector' in df_portfolio.columns:
            sector_exposure = df_portfolio.groupby('sector')['weight'].sum()
            max_sector = sector_exposure.max()
            risk_checks['sector_exposure_ok'] = max_sector <= self.config['max_sector_exposure']
            
            if not risk_checks['sector_exposure_ok']:
                logger.warning(f"行业暴露{max_sector:.2%}超过限制{self.config['max_sector_exposure']:.2%}")
        
        # 检查总仓位
        total_position = df_portfolio['weight'].sum()
        risk_checks['total_position_ok'] = total_position <= 1.0
        
        if not risk_checks['total_position_ok']:
            logger.warning(f"总仓位{total_position:.2%}超过100%")
        
        return risk_checks
    
    def check_drawdown(self, current_value: float) -> bool:
        """
        检查回撤是否超过限制
        
        Args:
            current_value: 当前组合价值
            
        Returns:
            是否触发回撤控制
        """
        if not self.config['enable_drawdown_control']:
            return False
        
        # 更新运行时状态
        self.portfolio_value = current_value
        self.peak_value = max(self.peak_value, current_value)
        self.current_drawdown = (self.portfolio_value - self.peak_value) / self.peak_value
        
        # 检查是否超过最大回撤
        if self.current_drawdown < self.config['max_drawdown']:
            logger.warning(f"当前回撤{self.current_drawdown:.2%}超过限制{self.config['max_drawdown']:.2%}")
            return True
        
        return False
    
    def adjust_weights(self, df_portfolio: pd.DataFrame) -> pd.DataFrame:
        """
        调整组合权重以满足风险限制
        
        Args:
            df_portfolio: 原始组合数据
            
        Returns:
            调整后的组合数据
        """
        df_adjusted = df_portfolio.copy()
        if df_adjusted.empty or 'weight' not in df_adjusted.columns:
            return df_adjusted

        gross_before = float(df_adjusted['weight'].sum())
        if gross_before <= 0:
            return df_adjusted
        
        # 调整单只股票仓位
        if self.config['enable_position_limit']:
            max_weight = self.config['max_single_position']
            clipped_weight = df_adjusted['weight'].clip(upper=max_weight)
            total_weight = float(clipped_weight.sum())
            preserve_gross_exposure = self.config.get('preserve_gross_exposure', True)

            if total_weight > 0 and preserve_gross_exposure:
                target_gross = min(gross_before, 1.0)
                if total_weight > target_gross:
                    clipped_weight = clipped_weight * (target_gross / total_weight)
                elif total_weight < target_gross:
                    # 仅在未触碰单票上限的标的上补仓，避免放大后再次突破单票上限
                    headroom = (max_weight - clipped_weight).clip(lower=0.0)
                    total_headroom = float(headroom.sum())
                    needed = target_gross - total_weight
                    if total_headroom > 0 and needed > 0:
                        add_ratio = min(1.0, needed / total_headroom)
                        clipped_weight = clipped_weight + headroom * add_ratio
                    if float(clipped_weight.sum()) + 1e-12 < target_gross:
                        logger.warning(
                            "受单票上限约束，无法恢复到原总仓位: target=%.2f%%, actual=%.2f%%",
                            target_gross * 100.0,
                            float(clipped_weight.sum()) * 100.0,
                        )
            elif total_weight > 0:
                clipped_weight = clipped_weight / total_weight

            df_adjusted['weight'] = clipped_weight
        
        # 调整行业暴露
        if 'sector' in df_adjusted.columns and self.config['max_sector_exposure'] < 1.0:
            sector_limit = self.config['max_sector_exposure']
            
            # TODO: 实现行业暴露调整逻辑
            # 这里简化处理，实际可能需要多次迭代调整
        
        logger.info(f"权重调整完成，总仓位: {df_adjusted['weight'].sum():.2%}")
        
        return df_adjusted
    
    def get_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        计算风险指标
        
        Args:
            returns: 收益率序列
            
        Returns:
            风险指标字典
        """
        metrics = {}
        
        # 年化波动率
        metrics['annual_volatility'] = returns.std() * np.sqrt(252)
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        metrics['max_drawdown'] = drawdown.min()
        
        # 下行波动率
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            metrics['downside_volatility'] = negative_returns.std() * np.sqrt(252)
        else:
            metrics['downside_volatility'] = 0.0
        
        # VaR（95%置信度）
        metrics['var_95'] = np.percentile(returns, 5)
        
        # CVaR（条件VaR）
        var_95 = metrics['var_95']
        metrics['cvar_95'] = returns[returns <= var_95].mean()
        
        return metrics


if __name__ == "__main__":
    # 测试风险控制
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    np.random.seed(42)
    stock_codes = ['sh.600000', 'sh.600004', 'sh.600006', 'sh.600007', 'sh.600008']
    
    data = []
    for i, code in enumerate(stock_codes):
        data.append({
            'code': code,
            'weight': np.random.uniform(0.02, 0.1),
            'sector': ['金融', '金融', '科技', '科技', '消费'][i]
        })
    
    df_portfolio = pd.DataFrame(data)
    
    # 测试风险控制
    print("测试风险控制...")
    risk_control = RiskControl()
    
    # 测试组合风险检查
    print("\n组合风险检查:")
    risk_checks = risk_control.check_portfolio_risk(df_portfolio)
    for check, result in risk_checks.items():
        print(f"  {check}: {'通过' if result else '未通过'}")
    
    # 测试权重调整
    print("\n调整前权重:")
    print(df_portfolio[['code', 'sector', 'weight']])
    
    df_adjusted = risk_control.adjust_weights(df_portfolio)
    print("\n调整后权重:")
    print(df_adjusted[['code', 'sector', 'weight']])
    
    # 测试回撤检查
    print("\n测试回撤检查:")
    for value in [1.0, 0.95, 0.90, 0.85, 0.80]:
        triggered = risk_control.check_drawdown(value)
        print(f"  组合价值: {value:.2f}, 回撤: {risk_control.current_drawdown:.2%}, 触发控制: {triggered}")
