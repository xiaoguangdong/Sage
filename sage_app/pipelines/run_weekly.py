"""
主入口：每周运行流程
"""
import pandas as pd
import numpy as np
import yaml
import logging
import json
import sys
from pathlib import Path
from datetime import datetime

# 导入项目模块
sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.data._shared.runtime import get_data_path, get_tushare_root
from scripts.strategy.build_industry_signal_contract import build_industry_signal_contract_artifacts
from sage_app.data.data_loader import DataLoader
from sage_app.pipelines.overlay_config import resolve_industry_overlay_config
from sage_core.utils.column_normalizer import normalize_security_columns
from sage_core.utils.logging_utils import setup_logging
from sage_core.data.universe import Universe
from sage_core.features.price_features import PriceFeatures
from sage_core.features.market_features import MarketFeatures
from sage_core.trend.trend_model import create_trend_model
try:
    from sage_core.stock_selection.rank_model import RankModelLGBM
except ModuleNotFoundError:  # lightgbm 未安装时允许 weekly 链路继续运行
    RankModelLGBM = None
from sage_core.execution.entry_model import EntryModelLR
from sage_core.execution.signal_contract import (
    apply_industry_overlay,
    build_stock_industry_map_from_features,
    build_stock_signal_contract,
    select_champion_signals,
)
from sage_core.execution.unified_signal_contract import build_unified_signal_contract
from sage_core.governance.strategy_governance import (
    ChampionChallengerEngine,
    ChallengerConfig,
    MultiAlphaChallengerStrategies,
    SeedBalanceStrategy,
    StrategyGovernanceConfig,
    decide_auto_promotion,
    normalize_strategy_id,
    save_strategy_outputs,
)
from sage_core.stock_selection.stock_selector import SelectionConfig
from sage_core.portfolio.construction import PortfolioConstruction
from sage_core.portfolio.risk_control import RiskControl

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
        new_config_dir = Path(__file__).resolve().parents[2] / "config" / "app"
        legacy_config_dir = Path(__file__).resolve().parents[1] / "config"
        config_dir = str(new_config_dir if new_config_dir.exists() else legacy_config_dir)
    
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
        logger.warning("Baostock raw 数据为空，回退读取 Tushare 日线/基础数据")
        df = _load_tushare_fallback()
    
    if df is None or len(df) == 0:
        logger.error("无法加载数据")
        return None

    df = normalize_security_columns(df, inplace=False)
    
    logger.info(f"加载数据完成，共{len(df)}条记录")
    
    # 检查数据质量
    quality_report = loader.check_data_quality(df)
    missing_values = quality_report.get("missing_values", {})
    missing_nonzero = {k: int(v) for k, v in missing_values.items() if int(v) > 0}
    date_range = quality_report.get("date_range")
    stock_count = len(quality_report.get("stock_codes", []))
    logger.info(
        "数据质量报告: rows=%s, stock_count=%s, date_range=%s, missing_nonzero=%s",
        quality_report.get("total_rows"),
        stock_count,
        date_range,
        missing_nonzero,
    )
    
    return df


def _load_tushare_fallback(years: int = 2) -> pd.DataFrame:
    """
    当 raw/baostock 不可用时，使用 data/tushare 作为兜底输入。
    """
    tushare_root = get_tushare_root()
    daily_dir = tushare_root / "daily"
    daily_files = sorted(daily_dir.glob("daily_*.parquet"))
    if not daily_files:
        logger.warning("未找到 Tushare 日线文件: %s", daily_dir)
        return pd.DataFrame()

    selected = daily_files[-max(1, int(years)) :]
    frames = []
    for path in selected:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            logger.warning("读取日线失败: %s (%s)", path, exc)
    if not frames:
        return pd.DataFrame()

    daily = pd.concat(frames, ignore_index=True)
    rename_map = {"vol": "volume"}
    for old, new in rename_map.items():
        if old in daily.columns and new not in daily.columns:
            daily[new] = daily[old]

    basic_path = tushare_root / "daily_basic_all.parquet"
    if basic_path.exists():
        try:
            import pyarrow.parquet as pq

            basic_cols = ["ts_code", "trade_date", "turnover_rate_f", "total_mv"]
            available_cols = set(pq.read_schema(basic_path).names)
            selected_cols = [c for c in basic_cols if c in available_cols]
            if not {"ts_code", "trade_date"}.issubset(set(selected_cols)):
                selected_cols = []
            basic = pd.read_parquet(basic_path, columns=selected_cols) if selected_cols else pd.DataFrame()
            if "turnover_rate_f" in basic.columns and "turnover" not in basic.columns:
                basic["turnover"] = pd.to_numeric(basic["turnover_rate_f"], errors="coerce")
            if "total_mv" in basic.columns and "market_cap" not in basic.columns:
                basic["market_cap"] = pd.to_numeric(basic["total_mv"], errors="coerce")
            keep_cols = [c for c in ["ts_code", "trade_date", "turnover", "market_cap"] if c in basic.columns]
            if len(keep_cols) >= 2:
                daily = daily.merge(basic[keep_cols], on=["ts_code", "trade_date"], how="left")
        except Exception as exc:
            logger.warning("读取/合并 daily_basic_all 失败: %s", exc)

    logger.info("Tushare fallback 加载完成: rows=%d, files=%d", len(daily), len(selected))
    return daily


def _load_hs300_index_frame(reference_trade_date: str | None = None) -> pd.DataFrame:
    """
    加载沪深300指数行情（优先 data/tushare/index/）。
    返回列至少包含: trade_date, close, pct_chg
    """
    tushare_root = get_tushare_root()
    candidates = [
        tushare_root / "index" / "index_000300_SH_ohlc.parquet",
        tushare_root / "index" / "index_ohlc_all.parquet",
        tushare_root / "index_000300_SH_ohlc.parquet",
        tushare_root / "index_ohlc_all.parquet",
    ]
    index_df = pd.DataFrame()
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.warning("读取指数文件失败: %s (%s)", path, exc)
            continue
        if "code" in df.columns:
            df = df[df["code"].astype(str) == "000300.SH"].copy()
        if df.empty:
            continue
        index_df = df
        break

    if index_df.empty:
        return index_df

    if "trade_date" not in index_df.columns:
        if "date" in index_df.columns:
            index_df["trade_date"] = index_df["date"]
        elif "datetime" in index_df.columns:
            index_df["trade_date"] = index_df["datetime"]
    index_df["trade_date"] = pd.to_datetime(index_df["trade_date"], errors="coerce")
    index_df = index_df.dropna(subset=["trade_date", "close"]).copy()
    index_df["trade_date"] = index_df["trade_date"].dt.strftime("%Y%m%d")

    if "pct_chg" not in index_df.columns:
        if "pct_change" in index_df.columns:
            index_df["pct_chg"] = pd.to_numeric(index_df["pct_change"], errors="coerce")
        else:
            index_df["pct_chg"] = pd.to_numeric(index_df["close"], errors="coerce").pct_change() * 100.0

    if reference_trade_date:
        ref = str(reference_trade_date)
        index_df = index_df[index_df["trade_date"] <= ref].copy()

    index_df = index_df.sort_values("trade_date").reset_index(drop=True)
    return index_df


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
            "satellite_strategy_v1",
        ])),
    )

    challenger_config = ChallengerConfig(
        positive_growth_weight=float(challenger_weights.get("positive_growth_weight", 0.7)),
        positive_frontier_weight=float(challenger_weights.get("positive_frontier_weight", 0.3)),
        satellite_growth_weight=float(challenger_weights.get("satellite_growth_weight", 0.35)),
        satellite_frontier_weight=float(challenger_weights.get("satellite_frontier_weight", 0.25)),
        satellite_rps_weight=float(challenger_weights.get("satellite_rps_weight", 0.20)),
        satellite_elasticity_weight=float(challenger_weights.get("satellite_elasticity_weight", 0.10)),
        satellite_not_priced_weight=float(challenger_weights.get("satellite_not_priced_weight", 0.10)),
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


def _signals_to_ranked(execution_signals: pd.DataFrame) -> pd.DataFrame:
    if execution_signals is None or execution_signals.empty:
        return pd.DataFrame()
    df_ranked = execution_signals.copy()
    score_col = "score_final" if "score_final" in df_ranked.columns else "score"
    df_ranked = df_ranked.rename(columns={"ts_code": "code", score_col: "rank_score"}).copy()
    if "rank" not in df_ranked.columns:
        df_ranked["rank"] = df_ranked["rank_score"].rank(ascending=False, method="first")
    if "industry_l1" in df_ranked.columns and "sector" not in df_ranked.columns:
        df_ranked["sector"] = df_ranked["industry_l1"]
    return df_ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)


def _load_evaluation_frame(evaluation_path: Path) -> pd.DataFrame:
    if not evaluation_path.exists():
        return pd.DataFrame()
    if evaluation_path.suffix.lower() == ".parquet":
        return pd.read_parquet(evaluation_path)
    if evaluation_path.suffix.lower() == ".json":
        return pd.read_json(evaluation_path)
    if evaluation_path.suffix.lower() == ".jsonl":
        return pd.read_json(evaluation_path, lines=True)
    if evaluation_path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(evaluation_path)
    logger.warning("不支持的评估文件格式: %s", evaluation_path)
    return pd.DataFrame()


def _append_decision_jsonl(path: Path, decision: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False) + "\n")


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _build_and_load_industry_snapshot(
    latest_trade_date: str,
    lookback_days: dict[str, int] | None = None,
    default_lookback_days: int = 30,
    half_life_days: dict[str, float] | None = None,
    default_half_life_days: float = 7.0,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    output_dir = get_data_path("signals", "industry", ensure=True)
    result = build_industry_signal_contract_artifacts(
        output_dir=output_dir,
        as_of_date=latest_trade_date,
        signal_lookback_days=lookback_days or {},
        default_lookback_days=int(default_lookback_days),
        signal_half_life_days=half_life_days or {},
        default_half_life_days=float(default_half_life_days),
    )
    snapshot_path = Path(result.get("signal_snapshot_path", output_dir / "industry_signal_snapshot_latest.parquet"))
    score_snapshot_path = Path(result.get("score_snapshot_path", output_dir / "industry_score_snapshot_latest.parquet"))
    if not result.get("generated", False):
        if snapshot_path.exists():
            logger.warning("行业信号未重建，回退读取已有快照: %s", snapshot_path)
            score_snapshot = pd.read_parquet(score_snapshot_path) if score_snapshot_path.exists() else pd.DataFrame()
            return pd.read_parquet(snapshot_path), score_snapshot, snapshot_path, score_snapshot_path
        logger.warning("行业信号未重建且无历史快照可用")
        return pd.DataFrame(), pd.DataFrame(), snapshot_path, score_snapshot_path

    summary = result.get("summary", {}) or {}
    freshness = summary.get("signal_freshness", {})
    logger.info(
        "行业信号已对齐: as_of=%s, rows=%s, freshness=%s",
        summary.get("as_of_date"),
        summary.get("rows_signal_snapshot"),
        freshness,
    )
    snapshot = pd.read_parquet(snapshot_path) if snapshot_path.exists() else pd.DataFrame()
    score_snapshot = pd.read_parquet(score_snapshot_path) if score_snapshot_path.exists() else pd.DataFrame()
    return snapshot, score_snapshot, snapshot_path, score_snapshot_path


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

    latest_trade_date_hint = None
    if "trade_date" in df_features.columns:
        latest_trade_date_hint = pd.to_datetime(df_features["trade_date"].max(), errors="coerce").strftime("%Y%m%d")

    index_candidates = {'000300.SH', 'sh.000300', '000300.SZ'}
    index_code = next((c for c in index_candidates if c in set(df_features['code'].values)), None)
    df_index = df_features[df_features['code'] == index_code].copy() if index_code else pd.DataFrame()

    if df_index.empty:
        df_index = _load_hs300_index_frame(reference_trade_date=latest_trade_date_hint)
        if not df_index.empty:
            logger.info("已从Tushare指数文件加载沪深300: rows=%d", len(df_index))

    if len(df_index) > 0:
        trend_result = trend_model.predict(df_index)
        trend_state = trend_result['state']
        logger.info(f"趋势状态: {trend_result['state_name']} (state={trend_state})")
    else:
        trend_result = {'position_suggestion': 0.3, 'state_name': '震荡'}
        trend_state = 1  # 默认震荡
        logger.warning("无法获取沪深300指数数据，使用默认趋势状态: 震荡")
    
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

    governance_cfg = (config.get("strategy_governance") or {})
    governance_raw = governance_cfg.get("strategy_governance", {})
    auto_cfg = governance_cfg.get("auto_promotion", {})
    configured_champion = normalize_strategy_id(
        governance_raw.get("active_champion_id", governance_engine.governance_config.active_champion_id)
    )
    manual_mode = governance_raw.get("champion_source", "manual") == "manual"
    auto_enabled = bool(auto_cfg.get("enabled", False))
    promotion_decision = {
        "enabled": auto_enabled,
        "current_champion": configured_champion,
        "next_champion": configured_champion,
        "promoted": False,
        "reason": "disabled",
    }

    if auto_enabled:
        evaluation_path = Path(auto_cfg.get("evaluation_path", "data/backtest/governance/shadow_eval.parquet"))
        try:
            evaluation_df = _load_evaluation_frame(evaluation_path)
            promotion_decision = decide_auto_promotion(
                evaluation_df=evaluation_df,
                current_champion=configured_champion,
                challengers=governance_engine.governance_config.normalized_challengers(),
                as_of_date=latest_trade_date,
                enabled=auto_enabled,
                manual_mode=manual_mode,
                allow_when_manual=bool(auto_cfg.get("allow_when_manual", False)),
                consecutive_periods=int(auto_cfg.get("consecutive_periods", 3)),
                max_drawdown_tolerance=float(auto_cfg.get("max_drawdown_tolerance", 0.02)),
                max_turnover_multiplier=float(auto_cfg.get("max_turnover_multiplier", 1.2)),
                min_cost_return_diff=float(auto_cfg.get("min_cost_return_diff", 0.0)),
                min_sharpe_diff=float(auto_cfg.get("min_sharpe_diff", 0.0)),
                min_data_quality=float(auto_cfg.get("min_data_quality", 0.95)),
            )
        except Exception as exc:
            logger.warning("自动晋升评估失败，回退手动冠军: %s", exc)
            promotion_decision = {
                "enabled": auto_enabled,
                "current_champion": configured_champion,
                "next_champion": configured_champion,
                "promoted": False,
                "reason": f"error:{exc}",
            }

    active_champion_id = normalize_strategy_id(
        promotion_decision.get("next_champion", configured_champion)
    )
    logger.info(
        "自动晋升决策: enabled=%s, promoted=%s, champion=%s -> %s, reason=%s",
        promotion_decision.get("enabled", False),
        promotion_decision.get("promoted", False),
        configured_champion,
        active_champion_id,
        promotion_decision.get("reason"),
    )
    decision_path = Path(auto_cfg.get("decision_path", "data/backtest/governance/promotion_decisions.jsonl"))
    decision_record = {
        "trade_date": latest_trade_date,
        "champion_source": governance_raw.get("champion_source", "manual"),
        **promotion_decision,
    }
    _append_decision_jsonl(decision_path, decision_record)

    seed_data = df_features.copy()
    if 'ts_code' in seed_data.columns:
        seed_data = seed_data[~seed_data['ts_code'].isin(index_candidates)].copy()

    governance_output = governance_engine.run(
        trade_date=latest_trade_date,
        top_n=10,
        seed_data=seed_data,
        active_champion_id=active_champion_id,
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

    # 6. 统一信号契约：执行层只消费contract
    signal_contract = build_stock_signal_contract(
        trade_date=latest_trade_date,
        champion_id=active_champion_id,
        champion_signals=champion_signals,
        challenger_signals=challenger_signals,
        include_challengers=True,
    )
    contract_dir = get_data_path("signals", "stock_selector", "contracts", ensure=True)
    contract_path = contract_dir / f"stock_signal_contract_{latest_trade_date}.parquet"
    contract_latest = contract_dir / "stock_signal_contract_latest.parquet"
    signal_contract.to_parquet(contract_path, index=False)
    signal_contract.to_parquet(contract_latest, index=False)

    champion_contract = select_champion_signals(
        signal_contract,
        champion_id=active_champion_id,
        min_confidence=0.0,
    )
    strategy_cfg = config.get("strategy_governance", {}) if isinstance(config, dict) else {}
    industry_cfg = (strategy_cfg.get("industry_signals") or {}) if isinstance(strategy_cfg, dict) else {}
    lookback_days = industry_cfg.get("signal_lookback_days")
    half_life_days = industry_cfg.get("signal_half_life_days")
    default_lookback = industry_cfg.get("default_lookback_days", 30)
    default_half_life = industry_cfg.get("default_half_life_days", 7.0)
    overlay_resolved = resolve_industry_overlay_config(industry_cfg if isinstance(industry_cfg, dict) else None, trend_state)
    overlay_strength = float(overlay_resolved["overlay_strength"])
    mainline_strength = float(overlay_resolved["mainline_strength"])
    signal_weights = overlay_resolved["signal_weights"]
    tilt_strength = float(overlay_resolved.get("tilt_strength", 0.0))
    industry_tilts = overlay_resolved.get("industry_tilts", {})
    regime_name = overlay_resolved["regime_name"]
    logger.info(
        "行业叠加参数: regime=%s, overlay_strength=%.3f, mainline_strength=%.3f, tilt_strength=%.3f, signal_weights=%s, industry_tilt_count=%d",
        regime_name,
        overlay_strength,
        mainline_strength,
        tilt_strength,
        signal_weights,
        len(industry_tilts) if isinstance(industry_tilts, dict) else 0,
    )

    industry_snapshot, industry_score_snapshot, industry_snapshot_path, industry_score_snapshot_path = _build_and_load_industry_snapshot(
        latest_trade_date=latest_trade_date,
        lookback_days=lookback_days if isinstance(lookback_days, dict) else None,
        default_lookback_days=int(default_lookback),
        half_life_days=half_life_days if isinstance(half_life_days, dict) else None,
        default_half_life_days=float(default_half_life),
    )
    stock_industry_map = build_stock_industry_map_from_features(df_features)
    execution_signals = apply_industry_overlay(
        stock_signals=champion_contract,
        industry_snapshot=industry_snapshot,
        stock_industry_map=stock_industry_map,
        industry_score_snapshot=industry_score_snapshot,
        signal_weights=signal_weights,
        overlay_strength=overlay_strength,
        mainline_strength=mainline_strength,
        industry_tilts=industry_tilts if isinstance(industry_tilts, dict) else None,
        tilt_strength=tilt_strength,
    )
    exec_signal_path = contract_dir / f"execution_signals_{latest_trade_date}.parquet"
    execution_signals.to_parquet(exec_signal_path, index=False)
    logger.info(
        "执行信号契约已生成: contract=%s, execution=%s, rows=%d",
        contract_path,
        exec_signal_path,
        len(execution_signals),
    )

    unified_contract = build_unified_signal_contract(
        trade_date=latest_trade_date,
        stock_contract=signal_contract,
        industry_contract=industry_snapshot,
        trend_result=trend_result if isinstance(trend_result, dict) else None,
        include_challengers=True,
    )
    unified_contract_dir = get_data_path("signals", "contracts", ensure=True)
    unified_contract_path = unified_contract_dir / f"unified_signal_contract_{latest_trade_date}.parquet"
    unified_contract_latest_path = unified_contract_dir / "unified_signal_contract_latest.parquet"
    unified_contract.to_parquet(unified_contract_path, index=False)
    unified_contract.to_parquet(unified_contract_latest_path, index=False)
    logger.info(
        "统一信号契约已生成: path=%s, rows=%d",
        unified_contract_path,
        len(unified_contract),
    )

    # 7. Champion执行信号转组合候选
    df_ranked = _signals_to_ranked(execution_signals)
    if df_ranked.empty:
        logger.warning("Champion信号为空，回退到简单排序")
        df_ranked = df_features.copy()
        df_ranked['rank_score'] = np.random.rand(len(df_ranked))
        df_ranked['rank'] = df_ranked['rank_score'].rank(ascending=False)
        if 'code' not in df_ranked.columns and 'ts_code' in df_ranked.columns:
            df_ranked['code'] = df_ranked['ts_code']
    
    # 8. 构建组合
    logger.info("构建组合...")
    portfolio_constructor = PortfolioConstruction()
    portfolio = portfolio_constructor.construct_portfolio(df_ranked, trend_state)
    
    # 9. 风险控制
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
    portfolio_for_risk = portfolio.copy()
    if "sector" not in portfolio_for_risk.columns and "industry_l1" in portfolio_for_risk.columns:
        portfolio_for_risk["sector"] = portfolio_for_risk["industry_l1"]
    risk_checks_raw = risk_control.check_portfolio_risk(portfolio_for_risk) if len(portfolio_for_risk) > 0 else {}
    risk_checks = {k: bool(v) for k, v in risk_checks_raw.items()}
    logger.info(
        "仓位管理: market_vol=%.2f%%, trend_pos=%.2f%%, vol_target_pos=%.2f%%, final_pos=%.2f%%",
        (market_volatility * 100.0) if market_volatility is not None and not np.isnan(market_volatility) else float('nan'),
        trend_position * 100.0,
        position_info['final_position'] * 100.0,
        float(portfolio['weight'].sum()) * 100.0 if len(portfolio) > 0 else 0.0,
    )
    if risk_checks:
        logger.info("风控检查: %s", risk_checks)
    
    # 10. 输出结果
    logger.info("=" * 50)
    logger.info("组合构建完成")
    logger.info("=" * 50)
    logger.info(f"趋势状态: {trend_state}")
    logger.info(f"持仓数量: {len(portfolio)}")
    logger.info(f"总仓位: {portfolio['weight'].sum():.2%}")
    logger.info(f"\n前5只股票:")
    for i, row in portfolio.head(5).iterrows():
        logger.info(f"  {row['code']}: 权重 {row['weight']:.2%}, 排名 {row['rank']}")
    
    # 11. 保存结果
    output_dir = 'data/portfolio'
    Path(output_dir).mkdir(exist_ok=True)
    
    output_file = f"{output_dir}/portfolio_{datetime.now().strftime('%Y%m%d')}.csv"
    portfolio.to_csv(output_file, index=False)
    logger.info(f"组合结果已保存到: {output_file}")

    context_file = f"{output_dir}/execution_context_{datetime.now().strftime('%Y%m%d')}.json"
    context_payload = {
        "trade_date": latest_trade_date,
        "active_champion_id": active_champion_id,
        "signal_contract_path": str(contract_path),
        "execution_signal_path": str(exec_signal_path),
        "unified_signal_contract_path": str(unified_contract_path),
        "industry_snapshot_path": str(industry_snapshot_path) if industry_snapshot_path.exists() else None,
        "industry_score_snapshot_path": str(industry_score_snapshot_path) if industry_score_snapshot_path.exists() else None,
        "industry_overlay_config": {
            "regime": regime_name,
            "overlay_strength": overlay_strength,
            "mainline_strength": mainline_strength,
            "signal_weights": signal_weights,
        },
        "position_info": position_info,
        "target_exposure": float(target_exposure),
        "risk_checks": risk_checks,
    }
    context_payload = _json_safe(context_payload)
    with open(context_file, "w", encoding="utf-8") as f:
        json.dump(context_payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    logger.info(f"执行上下文已保存到: {context_file}")
    
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
    
    from sage_core.backtest.walk_forward import WalkForwardBacktest

    # 创建模型
    trend_model = create_trend_model(
        config.get('trend_model', {}).get('trend_model', {}).get('model_type', 'rule')
    )
    
    rank_model_config = config.get('rank_model', {})
    if rank_model_config.get('rank_model', {}).get('enabled', False):
        if RankModelLGBM is None:
            logger.warning("rank_model 启用但 lightgbm 不可用，回退为空模型")
            rank_model = None
        else:
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
