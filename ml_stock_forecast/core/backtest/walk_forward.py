"""
Walk-forward回测模块
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """Walk-forward回测类"""
    
    def __init__(self, config: dict = None):
        """
        初始化回测
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 默认配置
        self.default_config = {
            'train_weeks': 130,  # 训练周数（约2.5年）
            'test_weeks': 26,  # 测试周数（约半年）
            'roll_step': 26,  # 滚动步长（约半年）
            'initial_capital': 1000000,  # 初始资金
        }
        
        # 合并配置
        self.config = {**self.default_config, **self.config}
        
        # 回测结果
        self.results = {}
    
    def prepare_data_splits(self, df: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        准备时间划分
        
        Args:
            df: 股票数据DataFrame，包含'date'列
            
        Returns:
            时间划分列表，每个元素为(train_start, train_end, test_start, test_end)
        """
        # 获取时间范围
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # 转换为周数
        train_days = self.config['train_weeks'] * 7
        test_days = self.config['test_weeks'] * 7
        roll_days = self.config['roll_step'] * 7
        
        # 生成时间划分
        splits = []
        current_date = min_date
        
        while True:
            train_start = current_date
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days)
            
            # 检查是否超出数据范围
            if test_end > max_date:
                break
            
            splits.append((train_start, train_end, test_start, test_end))
            
            # 滚动到下一个窗口
            current_date += timedelta(days=roll_days)
        
        logger.info(f"生成{len(splits)}个时间划分")
        
        return splits
    
    def train_model(self, df_train: pd.DataFrame, trend_model, rank_model, entry_model):
        """
        训练模型
        
        Args:
            df_train: 训练数据
            trend_model: 趋势模型
            rank_model: 排序模型
            entry_model: 买卖点模型
        """
        logger.info(f"训练模型，数据范围: {df_train['date'].min()} 到 {df_train['date'].max()}")
        
        # 训练排序模型
        if rank_model is not None:
            # 创建排序标签
            labels = rank_model.create_ranking_label(df_train)
            df_train_clean = df_train.dropna()
            labels = labels.loc[df_train_clean.index]
            
            # 创建分组信息
            group_info = df_train_clean.groupby('date').size()
            
            # 训练
            rank_model.train(df_train_clean, labels, group_info)
            logger.info("排序模型训练完成")
        
        # 训练买卖点模型
        if entry_model is not None:
            # 创建买卖点标签
            labels = entry_model.create_entry_label(df_train)
            df_train_clean = df_train.dropna()
            labels = labels.loc[df_train_clean.index]
            
            # 训练
            entry_model.train(df_train_clean, labels)
            logger.info("买卖点模型训练完成")
    
    def run_backtest(self, df: pd.DataFrame, trend_model, rank_model, entry_model, 
                    portfolio_constructor, risk_control) -> Dict:
        """
        运行回测
        
        Args:
            df: 股票数据
            trend_model: 趋势模型
            rank_model: 排序模型
            entry_model: 买卖点模型
            portfolio_constructor: 组合构建器
            risk_control: 风险控制器
            
        Returns:
            回测结果字典
        """
        logger.info("开始Walk-forward回测")
        
        # 准备时间划分
        splits = self.prepare_data_splits(df)
        
        # 回测结果
        portfolio_returns = []
        portfolio_values = [self.config['initial_capital']]
        trade_records = []
        
        # 运行每个时间窗口
        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            logger.info(f"\n=== 窗口 {i+1}/{len(splits)} ===")
            logger.info(f"训练期: {train_start.date()} - {train_end.date()}")
            logger.info(f"测试期: {test_start.date()} - {test_end.date()}")
            
            # 获取训练数据
            df_train = df[(df['date'] >= train_start) & (df['date'] <= train_end)]
            
            # 训练模型
            self.train_model(df_train, trend_model, rank_model, entry_model)
            
            # 获取测试数据
            df_test = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
            
            # 运行测试
            window_returns, window_trades = self._run_test_window(
                df_test, trend_model, rank_model, entry_model, 
                portfolio_constructor, risk_control
            )
            
            # 记录结果
            portfolio_returns.extend(window_returns)
            trade_records.extend(window_trades)
            
            # 更新组合价值
            for ret in window_returns:
                portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
        # 计算回测指标
        metrics = self._calculate_metrics(portfolio_returns, portfolio_values)
        
        # 保存结果
        self.results = {
            'returns': portfolio_returns,
            'values': portfolio_values,
            'trades': trade_records,
            'metrics': metrics,
            'splits': splits
        }
        
        logger.info("回测完成")
        
        return self.results
    
    def _run_test_window(self, df_test: pd.DataFrame, trend_model, rank_model, entry_model,
                         portfolio_constructor, risk_control) -> Tuple[List[float], List[Dict]]:
        """
        运行单个测试窗口
        
        Args:
            df_test: 测试数据
            trend_model: 趋势模型
            rank_model: 排序模型
            entry_model: 买卖点模型
            portfolio_constructor: 组合构建器
            risk_control: 风险控制器
            
        Returns:
            (收益率列表, 交易记录列表)
        """
        returns = []
        trades = []
        
        # 获取测试日期
        dates = df_test['date'].unique()
        
        # 获取指数数据（用于趋势判断）
        df_index = df_test[df_test['code'] == '000300.SH'] if '000300.SH' in df_test['code'].values else None
        
        for i, date in enumerate(dates):
            # 获取当日数据
            df_day = df_test[df_test['date'] == date]
            
            # 预测趋势状态
            if trend_model is not None and df_index is not None:
                df_index_day = df_index[df_index['date'] <= date].tail(150)
                if len(df_index_day) > 0:
                    trend_result = trend_model.predict(df_index_day)
                    trend_state = trend_result['state']
                else:
                    trend_state = 1  # 默认震荡
            else:
                trend_state = 1  # 默认震荡
            
            # 排序股票
            if rank_model is not None and rank_model.is_trained:
                df_ranked = rank_model.predict(df_day)
            else:
                # 如果没有排序模型，随机排序
                df_ranked = df_day.copy()
                df_ranked['rank_score'] = np.random.rand(len(df_ranked))
                df_ranked['rank'] = df_ranked['rank_score'].rank(ascending=False)
            
            # 构建组合
            portfolio = portfolio_constructor.construct_portfolio(df_ranked, trend_state)
            
            # 风险控制
            portfolio = risk_control.adjust_weights(portfolio)
            
            # 计算组合收益
            if len(portfolio) > 0:
                # 计算当日收益率
                portfolio_return = self._calculate_daily_return(portfolio, df_test, date)
                returns.append(portfolio_return)
                
                # 记录交易
                trades.append({
                    'date': date,
                    'trend_state': trend_state,
                    'num_positions': len(portfolio),
                    'total_weight': portfolio['weight'].sum(),
                    'return': portfolio_return,
                    'top_stocks': portfolio.nlargest(3, 'rank_score')['code'].tolist()
                })
            else:
                # 空仓
                returns.append(0.0)
                trades.append({
                    'date': date,
                    'trend_state': trend_state,
                    'num_positions': 0,
                    'total_weight': 0.0,
                    'return': 0.0,
                    'top_stocks': []
                })
        
        return returns, trades
    
    def _calculate_daily_return(self, portfolio: pd.DataFrame, df_test: pd.DataFrame, date) -> float:
        """
        计算当日组合收益
        
        Args:
            portfolio: 组合数据
            df_test: 测试期数据（包含历史）
            date: 当前日期
            
        Returns:
            组合收益率
        """
        if 'code' not in df_test.columns and 'stock' in df_test.columns:
            df_test = df_test.rename(columns={'stock': 'code'})

        df_hist = df_test[df_test['date'] <= date]
        portfolio_return = 0.0
        total_weight = portfolio['weight'].sum()

        for _, row in portfolio.iterrows():
            code = row['code']
            weight = row['weight']
            hist = df_hist[df_hist['code'] == code].sort_values('date')
            if len(hist) < 2:
                continue
            prev_close = hist.iloc[-2]['close']
            curr_close = hist.iloc[-1]['close']
            if prev_close and prev_close != 0:
                ret = (curr_close - prev_close) / prev_close
                portfolio_return += weight * ret

        if total_weight > 0:
            portfolio_return = portfolio_return / total_weight

        return portfolio_return
    
    def _calculate_metrics(self, returns: List[float], values: List[float]) -> Dict[str, float]:
        """
        计算回测指标
        
        Args:
            returns: 收益率列表
            values: 组合价值列表
            
        Returns:
            指标字典
        """
        metrics = {}
        
        returns = pd.Series(returns)
        
        # 累计收益
        metrics['total_return'] = (values[-1] / values[0] - 1)
        
        # 年化收益
        n_days = len(returns)
        years = n_days / 252
        metrics['annual_return'] = (values[-1] / values[0]) ** (1 / years) - 1
        
        # 年化波动率
        metrics['annual_volatility'] = returns.std() * np.sqrt(252)
        
        # 夏普比率
        risk_free_rate = 0.03  # 假设无风险利率为3%
        metrics['sharpe_ratio'] = (metrics['annual_return'] - risk_free_rate) / metrics['annual_volatility']
        
        # 最大回撤
        cumulative = pd.Series(values) / values[0]
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        metrics['max_drawdown'] = drawdown.min()
        
        # 胜率
        metrics['win_rate'] = (returns > 0).mean()
        
        # 盈亏比
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        if len(losing_returns) > 0:
            metrics['profit_loss_ratio'] = winning_returns.mean() / abs(losing_returns.mean())
        else:
            metrics['profit_loss_ratio'] = np.inf
        
        return metrics


if __name__ == "__main__":
    # 测试回测
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2020-12-31', freq='D')
    stock_codes = ['sh.600000', 'sh.600004', 'sh.600006', 'sh.600007', 'sh.600008']
    
    all_data = []
    for code in stock_codes:
        close = 10 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        turnover = np.random.uniform(0.01, 0.1, len(dates))
        
        for i, date in enumerate(dates):
            all_data.append({
                'date': date,
                'code': code,
                'close': close[i],
                'turnover': turnover[i]
            })
    
    df = pd.DataFrame(all_data)
    
    # 测试回测
    print("测试Walk-forward回测...")
    
    from models.trend_model import TrendModelRule
    from models.entry_model import EntryModelLR
    from portfolio.construction import PortfolioConstruction
    from portfolio.risk_control import RiskControl
    
    backtest = WalkForwardBacktest()
    
    # 创建模型
    trend_model = TrendModelRule()
    rank_model = None  # 简化测试，不使用排序模型
    entry_model = EntryModelLR()
    portfolio_constructor = PortfolioConstruction()
    risk_control = RiskControl()
    
    # 运行回测
    results = backtest.run_backtest(df, trend_model, rank_model, entry_model, 
                                    portfolio_constructor, risk_control)
    
    print(f"\n回测结果:")
    print(f"总收益: {results['metrics']['total_return']:.2%}")
    print(f"年化收益: {results['metrics']['annual_return']:.2%}")
    print(f"年化波动: {results['metrics']['annual_volatility']:.2%}")
    print(f"夏普比率: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['metrics']['max_drawdown']:.2%}")
    print(f"胜率: {results['metrics']['win_rate']:.2%}")
