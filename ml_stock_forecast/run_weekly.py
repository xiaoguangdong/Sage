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
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_loader import DataLoader
from utils.column_normalizer import normalize_security_columns
from data.universe import Universe
from features.price_features import PriceFeatures
from features.market_features import MarketFeatures
from models.trend_model import create_trend_model
from models.rank_model import RankModelLGBM
from models.entry_model import EntryModelLR
from portfolio.construction import PortfolioConstruction
from portfolio.risk_control import RiskControl
from backtest.walk_forward import WalkForwardBacktest

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_dir: str = 'config') -> dict:
    """
    加载配置文件
    
    Args:
        config_dir: 配置文件目录
        
    Returns:
        配置字典
    """
    config = {}
    
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
        trend_state = 1  # 默认震荡
        logger.warning("无法获取指数数据，使用默认趋势状态: 震荡")
    
    # 5. 训练和预测（简化处理，实际需要使用历史数据训练）
    if rank_model is not None:
        logger.info("训练排序模型...")
        # TODO: 实现训练逻辑
    
    if entry_model is not None:
        logger.info("训练买卖点模型...")
        # TODO: 实现训练逻辑
    
    # 6. 排序股票
    if rank_model is not None and rank_model.is_trained:
        logger.info("排序股票...")
        df_ranked = rank_model.predict(df_features)
    else:
        # 如果没有排序模型，使用简单排序
        logger.info("使用简单排序...")
        df_ranked = df_features.copy()
        df_ranked['rank_score'] = np.random.rand(len(df_ranked))
        df_ranked['rank'] = df_ranked['rank_score'].rank(ascending=False)
    
    # 7. 构建组合
    logger.info("构建组合...")
    portfolio_constructor = PortfolioConstruction()
    portfolio = portfolio_constructor.construct_portfolio(df_ranked, trend_state)
    
    # 8. 风险控制
    logger.info("风险控制...")
    risk_control = RiskControl()
    portfolio = risk_control.adjust_weights(portfolio)
    
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
    output_dir = 'data/processed'
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
