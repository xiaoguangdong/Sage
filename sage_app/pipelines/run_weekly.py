"""
主入口：每周运行流程
"""
import pandas as pd
import numpy as np
import yaml
import logging
import sys
from pathlib import Path
from datetime import datetime

# 导入项目模块
sys.path.append(str(Path(__file__).resolve().parents[2]))

from sage_app.data.data_loader import DataLoader
from sage_core.utils.column_normalizer import normalize_security_columns
from sage_core.utils.logging_utils import setup_logging
from sage_core.data.universe import Universe
from sage_core.features.price_features import PriceFeatures
from sage_core.features.market_features import MarketFeatures
from sage_core.models.trend_model import create_trend_model
from sage_core.models.rank_model import RankModelLGBM
from sage_core.models.entry_model import EntryModelLR
from sage_core.models.strategy_governance import (
    ChampionChallengerEngine,
    ChallengerConfig,
    MultiAlphaChallengerStrategies,
    SeedBalanceStrategy,
    StrategyGovernanceConfig,
    normalize_strategy_id,
    save_strategy_outputs,
)
from sage_core.models.stock_selector import SelectionConfig
from sage_core.portfolio.construction import PortfolioConstruction
from sage_core.portfolio.risk_control import RiskControl
from sage_core.backtest.walk_forward import WalkForwardBacktest

# 配置日志
log_path = setup_logging("weekly")
logger = logging.getLogger(__name__)
logger.info(f"日志文件: {log_path}")


def load_config(config_dir: str | None = None) -> dict:
    """
    加载配置文件
    
    Args:
        config_dir: 配置文件目录
        
    Returns:
        配置字典
    """
    config = {}

    if config_dir is None:
        config_dir = str(Path(__file__).resolve().parents[1] / "config")
    
    # 加载趋势模型配置
    try:
        with open(f'{config_dir}/trend_model.yaml', 'r', encoding='utf-8') as f:
            config['trend_model'] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载趋势模型配置: {e}")
    
    # 加载排序模型配置
    try:
        with open(f'{config_dir}/rank_model.yaml', 'r', encoding='utf-8') as f:
            config['rank_model'] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载排序模型配置: {e}")
    
    # 加载买卖点模型配置
    try:
        with open(f'{config_dir}/entry_model.yaml', 'r', encoding='utf-8') as f:
            config['entry_model'] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载买卖点模型配置: {e}")
    
    # 加载风险控制配置
    try:
        with open(f'{config_dir}/risk_control.yaml', 'r', encoding='utf-8') as f:
            config['risk_control'] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载风险控制配置: {e}")

    # 加载策略治理配置
    try:
        with open(f'{config_dir}/strategy_governance.yaml', 'r', encoding='utf-8') as f:
            config['strategy_governance'] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载策略治理配置: {e}")
    
    return config


def load_data(data_dir: str = 'data') -> pd.DataFrame:
    """
    加载数据
    
    Args:
        data_dir: 数据目录
        
    Returns:
        股票数据DataFrame
    """
    logger.info("加载数据...")
    
    # 使用数据加载器
    loader = DataLoader(data_dir)
    
    # 加载所有Baostock数据（默认读取raw目录全部parquet）
    df = loader.load_all_baostock_data()
    
    if df is None or len(df) == 0:
        logger.error("无法加载数据")
        return None

    df = normalize_security_columns(df, inplace=False)
    
    logger.info(f"加载数据完成，共{len(df)}条记录")
    
    # 检查数据质量
    quality_report = loader.check_data_quality(df)
    logger.info(f"数据质量报告: {quality_report}")
    
    return df


def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    """
    过滤股票池
    
    Args:
        df: 股票数据
        
    Returns:
        过滤后的股票数据
    """
    logger.info("过滤股票池...")
    
    universe = Universe()
    df_filtered = universe.filter_stocks(
        df,
        exclude_st=True,
        exclude_suspended=True,
        min_turnover=0.01,
        min_market_cap=10  # 10亿市值
    )
    
    logger.info(f"过滤后股票数量: {len(df_filtered)}")
    
    return df_filtered


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算特征
    
    Args:
        df: 股票数据
        
    Returns:
        包含特征的DataFrame
    """
    logger.info("计算特征...")
    
    # 价格特征
    price_features = PriceFeatures()
    df = price_features.calculate_all_features(df)
    
    # 市场特征（基于沪深300）
    market_features = MarketFeatures(index_code="000300.SH")
    # 注意：市场特征需要单独计算，这里简化处理
    
    logger.info("特征计算完成")
    
    return df


def _build_governance_engine(config: dict) -> ChampionChallengerEngine:
    governance_cfg = (config.get("strategy_governance") or {})
    governance_raw = governance_cfg.get("strategy_governance", {})
    challenger_weights = governance_cfg.get("challenger_weights", {})
    seed_raw = governance_cfg.get("seed_balance_strategy", {})

    active_champion_id = normalize_strategy_id(
        governance_raw.get("active_champion_id", "seed_balance_strategy")
    )
    governance = StrategyGovernanceConfig(
        active_champion_id=active_champion_id,
        champion_source=governance_raw.get("champion_source", "manual"),
        manual_effective_date=governance_raw.get("manual_effective_date"),
        manual_reason=governance_raw.get("manual_reason"),
        challengers=tuple(governance_raw.get("challengers", [
            "balance_strategy_v1",
            "positive_strategy_v1",
            "value_strategy_v1",
        ])),
    )

    challenger_config = ChallengerConfig(
        positive_growth_weight=float(challenger_weights.get("positive_growth_weight", 0.7)),
        positive_frontier_weight=float(challenger_weights.get("positive_frontier_weight", 0.3)),
    )
    selector_config = SelectionConfig(
        model_type=seed_raw.get("model_type", "lgbm"),
        label_horizons=tuple(seed_raw.get("label_horizons", [20, 60, 120])),
        label_weights=tuple(seed_raw.get("label_weights", [0.5, 0.3, 0.2])),
        risk_adjusted=bool(seed_raw.get("risk_adjusted", True)),
        industry_col=seed_raw.get("industry_col", "industry_l1"),
    )

    seed_strategy = SeedBalanceStrategy(selector_config=selector_config)
    challenger_strategies = MultiAlphaChallengerStrategies(config=challenger_config)
    return ChampionChallengerEngine(
        governance_config=governance,
        seed_strategy=seed_strategy,
        challenger_strategies=challenger_strategies,
    )


def _signals_to_ranked(champion_signals: pd.DataFrame) -> pd.DataFrame:
    if champion_signals is None or champion_signals.empty:
        return pd.DataFrame()
    df_ranked = champion_signals.rename(
        columns={
            "ts_code": "code",
            "score": "rank_score",
        }
    ).copy()
    if "rank" not in df_ranked.columns:
        df_ranked["rank"] = df_ranked["rank_score"].rank(ascending=False, method="first")
    return df_ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)


def run_weekly_workflow(config: dict, df: pd.DataFrame):
    """
    运行每周工作流
    
    Args:
        config: 配置字典
        df: 股票数据
    """
    logger.info("=" * 50)
    logger.info("开始每周工作流")
    logger.info("=" * 50)
    
    # 1. 过滤股票池
    df_filtered = filter_universe(df)
    
    # 2. 计算特征
    df_features = calculate_features(df_filtered)
    
    # 3. 创建模型
    trend_model = create_trend_model(
        config.get('trend_model', {}).get('trend_model', {}).get('model_type', 'rule')
    )
    
    # 4. 预测趋势状态
    logger.info("预测趋势状态...")
    # 统一股票代码字段
    if 'code' not in df_features.columns and 'stock' in df_features.columns:
        df_features['code'] = df_features['stock']
    if 'stock' not in df_features.columns and 'code' in df_features.columns:
        df_features['stock'] = df_features['code']

    index_candidates = {'000300.SH', 'sh.000300', '000300.SZ'}
    index_code = next((c for c in index_candidates if c in set(df_features['code'].values)), None)
    df_index = df_features[df_features['code'] == index_code] if index_code else None
    
    if df_index is not None and len(df_index) > 0:
        trend_result = trend_model.predict(df_index)
        trend_state = trend_result['state']
        logger.info(f"趋势状态: {trend_result['state_name']} (state={trend_state})")
    else:
        trend_result = {'position_suggestion': 0.3, 'state_name': '震荡'}
        trend_state = 1  # 默认震荡
        logger.warning("无法获取指数数据，使用默认趋势状态: 震荡")
    
    # 5. 运行Champion/Challenger四策略，执行层只读取Champion
    logger.info("运行选股策略治理（Champion/Challenger）...")
    governance_engine = _build_governance_engine(config)

    if 'ts_code' not in df_features.columns and 'code' in df_features.columns:
        df_features['ts_code'] = df_features['code']
    if 'trade_date' not in df_features.columns and 'date' in df_features.columns:
        df_features['trade_date'] = df_features['date']

    if 'trade_date' in df_features.columns:
        latest_trade_date = pd.to_datetime(df_features['trade_date'].max(), errors='coerce').strftime('%Y%m%d')
    else:
        latest_trade_date = datetime.now().strftime('%Y%m%d')
    seed_data = df_features.copy()
    if 'ts_code' in seed_data.columns:
        seed_data = seed_data[~seed_data['ts_code'].isin(index_candidates)].copy()

    governance_output = governance_engine.run(
        trade_date=latest_trade_date,
        top_n=10,
        seed_data=seed_data,
        allocation_method="fixed",
        regime={0: "bear", 1: "sideways", 2: "bull"}.get(trend_state, "sideways"),
    )

    champion_signals = governance_output["champion_signals"]
    challenger_signals = governance_output["challenger_signals"]
    active_champion_id = governance_output["active_champion_id"]
    logger.info(
        "选股治理完成: active_champion=%s, champion_count=%d",
        active_champion_id,
        len(champion_signals),
    )

    signals_root = Path("data/signals/stock_selector")
    saved_paths = save_strategy_outputs(
        output_root=signals_root,
        trade_date=latest_trade_date,
        champion_id=active_champion_id,
        champion_signals=champion_signals,
        challenger_signals=challenger_signals,
    )
    logger.info("选股信号已保存: %s", {k: str(v) for k, v in saved_paths.items()})

    # 6. Champion信号转组合候选
    df_ranked = _signals_to_ranked(champion_signals)
    if df_ranked.empty:
        logger.warning("Champion信号为空，回退到简单排序")
        df_ranked = df_features.copy()
        df_ranked['rank_score'] = np.random.rand(len(df_ranked))
        df_ranked['rank'] = df_ranked['rank_score'].rank(ascending=False)
        if 'code' not in df_ranked.columns and 'ts_code' in df_ranked.columns:
            df_ranked['code'] = df_ranked['ts_code']
    
    # 7. 构建组合
    logger.info("构建组合...")
    portfolio_constructor = PortfolioConstruction()
    portfolio = portfolio_constructor.construct_portfolio(df_ranked, trend_state)
    
    # 8. 风险控制
    logger.info("风险控制...")
    risk_control = RiskControl((config.get('risk_control') or {}).get('risk_control'))
    portfolio = risk_control.adjust_weights(portfolio)

    market_volatility = risk_control.compute_market_volatility(df_index) if df_index is not None else np.nan
    latest_index_return = None
    latest_week_return = None
    if df_index is not None and len(df_index) > 0:
        df_index_sorted = df_index.sort_values('trade_date')
        if 'pct_chg' in df_index_sorted.columns:
            index_returns = pd.to_numeric(df_index_sorted['pct_chg'], errors='coerce') / 100.0
        elif 'return' in df_index_sorted.columns:
            index_returns = pd.to_numeric(df_index_sorted['return'], errors='coerce')
        elif 'close' in df_index_sorted.columns:
            index_returns = pd.to_numeric(df_index_sorted['close'], errors='coerce').pct_change()
        else:
            index_returns = pd.Series(dtype=float)
        index_returns = index_returns.dropna()
        if len(index_returns) >= 1:
            latest_index_return = float(index_returns.iloc[-1])
        if len(index_returns) >= 5:
            latest_week_return = float((1.0 + index_returns.tail(5)).prod() - 1.0)

    position_info = risk_control.compute_target_position(
        trend_state=trend_state,
        market_volatility=market_volatility,
        latest_index_return=latest_index_return,
        latest_week_return=latest_week_return,
        portfolio_drawdown=risk_control.current_drawdown,
    )
    trend_position = float(trend_result.get('position_suggestion', 1.0))
    target_exposure = min(position_info['final_position'], trend_position)
    portfolio = risk_control.scale_to_target_exposure(portfolio, target_exposure)
    logger.info(
        "仓位管理: market_vol=%.2f%%, trend_pos=%.2f%%, vol_target_pos=%.2f%%, final_pos=%.2f%%",
        (market_volatility * 100.0) if market_volatility is not None and not np.isnan(market_volatility) else float('nan'),
        trend_position * 100.0,
        position_info['final_position'] * 100.0,
        float(portfolio['weight'].sum()) * 100.0 if len(portfolio) > 0 else 0.0,
    )
    
    # 9. 输出结果
    logger.info("=" * 50)
    logger.info("组合构建完成")
    logger.info("=" * 50)
    logger.info(f"趋势状态: {trend_state}")
    logger.info(f"持仓数量: {len(portfolio)}")
    logger.info(f"总仓位: {portfolio['weight'].sum():.2%}")
    logger.info(f"\n前5只股票:")
    for i, row in portfolio.head(5).iterrows():
        logger.info(f"  {row['code']}: 权重 {row['weight']:.2%}, 排名 {row['rank']}")
    
    # 10. 保存结果
    output_dir = 'data/portfolio'
    Path(output_dir).mkdir(exist_ok=True)
    
    output_file = f"{output_dir}/portfolio_{datetime.now().strftime('%Y%m%d')}.csv"
    portfolio.to_csv(output_file, index=False)
    logger.info(f"组合结果已保存到: {output_file}")
    
    return portfolio


def run_backtest_workflow(config: dict, df: pd.DataFrame):
    """
    运行回测工作流
    
    Args:
        config: 配置字典
        df: 股票数据
    """
    logger.info("=" * 50)
    logger.info("开始回测工作流")
    logger.info("=" * 50)
    
    # 创建模型
    trend_model = create_trend_model(
        config.get('trend_model', {}).get('trend_model', {}).get('model_type', 'rule')
    )
    
    rank_model_config = config.get('rank_model', {})
    if rank_model_config.get('rank_model', {}).get('enabled', False):
        rank_model = RankModelLGBM(rank_model_config.get('lgbm_params', {}))
    else:
        rank_model = None
    
    entry_model_config = config.get('entry_model', {})
    if entry_model_config.get('entry_model', {}).get('enabled', False):
        entry_model = EntryModelLR(entry_model_config.get('entry_model', {}))
    else:
        entry_model = None
    
    # 创建组合构建器和风险控制器
    portfolio_constructor = PortfolioConstruction()
    risk_control = RiskControl()
    
    # 运行回测
    backtest = WalkForwardBacktest()
    results = backtest.run_backtest(
        df, trend_model, rank_model, entry_model,
        portfolio_constructor, risk_control
    )
    
    # 输出结果
    logger.info("=" * 50)
    logger.info("回测结果")
    logger.info("=" * 50)
    logger.info(f"总收益: {results['metrics']['total_return']:.2%}")
    logger.info(f"年化收益: {results['metrics']['annual_return']:.2%}")
    logger.info(f"年化波动: {results['metrics']['annual_volatility']:.2%}")
    logger.info(f"夏普比率: {results['metrics']['sharpe_ratio']:.2f}")
    logger.info(f"最大回撤: {results['metrics']['max_drawdown']:.2%}")
    logger.info(f"胜率: {results['metrics']['win_rate']:.2%}")
    logger.info(f"盈亏比: {results['metrics']['profit_loss_ratio']:.2f}")
    
    # 保存结果
    output_dir = 'data/processed'
    Path(output_dir).mkdir(exist_ok=True)
    
    output_file = f"{output_dir}/backtest_results_{datetime.now().strftime('%Y%m%d')}.csv"
    pd.DataFrame([results['metrics']]).to_csv(output_file, index=False)
    logger.info(f"回测结果已保存到: {output_file}")
    
    return results


def main():
    """主函数"""
    logger.info("程序启动")
    
    # 加载配置
    config = load_config()
    
    # 加载数据
    df = load_data()
    
    if df is None or len(df) == 0:
        logger.error("数据加载失败，程序退出")
        return
    
    # 选择运行模式
    mode = 'weekly'  # 默认每周运行模式
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    if mode == 'backtest':
        # 回测模式
        run_backtest_workflow(config, df)
    elif mode == 'weekly':
        # 每周运行模式
        run_weekly_workflow(config, df)
    else:
        logger.error(f"未知模式: {mode}")
        logger.info("使用方法: python run_weekly.py [weekly|backtest]")
    
    logger.info("程序结束")


if __name__ == "__main__":
    main()
