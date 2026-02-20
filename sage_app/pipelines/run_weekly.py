"""
主入口：每周运行流程
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 导入项目模块
sys.path.append(str(Path(__file__).resolve().parents[2]))

from sage_app.data.data_loader import DataLoader
from sage_app.pipelines.overlay_config import resolve_industry_overlay_config
from sage_core.data.universe import Universe
from sage_core.features.long_term_fundamental_features import LongTermFundamentalFeatures
from sage_core.features.market_features import MarketFeatures
from sage_core.features.price_features import PriceFeatures
from sage_core.trend.trend_model import create_trend_model
from sage_core.utils.column_normalizer import normalize_security_columns
from sage_core.utils.logging_utils import setup_logging
from scripts.data._shared.runtime import get_data_path, get_tushare_root
from scripts.strategy.build_industry_signal_contract import build_industry_signal_contract_artifacts

try:
    from sage_core.stock_selection.rank_model import RankModelLGBM
except ModuleNotFoundError:  # lightgbm 未安装时允许 weekly 链路继续运行
    RankModelLGBM = None
from sage_core.backtest.exposure import compute_factor_exposures
from sage_core.execution.entry_model import EntryModelLR
from sage_core.execution.signal_contract import (
    apply_industry_overlay,
    build_stock_industry_map_from_features,
    build_stock_signal_contract,
    select_champion_signals,
)
from sage_core.execution.unified_signal_contract import build_unified_signal_contract
from sage_core.governance.strategy_governance import (
    ChallengerConfig,
    ChampionChallengerEngine,
    MultiAlphaChallengerStrategies,
    SeedBalanceStrategy,
    StrategyGovernanceConfig,
    decide_auto_promotion,
    normalize_strategy_id,
    save_strategy_outputs,
)
from sage_core.portfolio.construction import PortfolioConstruction
from sage_core.portfolio.portfolio_manager import PortfolioManager
from sage_core.portfolio.risk_control import RiskControl
from sage_core.stock_selection.growth_stock_selector import GrowthStockSelector
from sage_core.stock_selection.hybrid_stock_selector import HybridStockSelector
from sage_core.stock_selection.stock_selector import SelectionConfig
from sage_core.stock_selection.value_stock_selector import ValueStockSelector

try:
    from sage_core.industry.macro_predictor import MacroPredictor
    from scripts.models.macro.export_macro_signals import export_macro_signals
except ImportError:
    MacroPredictor = None
    export_macro_signals = None

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
        with open(f"{config_dir}/trend_model.yaml", "r", encoding="utf-8") as f:
            config["trend_model"] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载趋势模型配置: {e}")

    # 加载排序模型配置
    try:
        with open(f"{config_dir}/rank_model.yaml", "r", encoding="utf-8") as f:
            config["rank_model"] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载排序模型配置: {e}")

    # 加载买卖点模型配置
    try:
        with open(f"{config_dir}/entry_model.yaml", "r", encoding="utf-8") as f:
            config["entry_model"] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载买卖点模型配置: {e}")

    # 加载风险控制配置
    try:
        with open(f"{config_dir}/risk_control.yaml", "r", encoding="utf-8") as f:
            config["risk_control"] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载风险控制配置: {e}")

    # 加载策略治理配置
    try:
        with open(f"{config_dir}/strategy_governance.yaml", "r", encoding="utf-8") as f:
            config["strategy_governance"] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载策略治理配置: {e}")

    # 加载归因配置
    try:
        with open(f"{config_dir}/attribution.yaml", "r", encoding="utf-8") as f:
            config["attribution"] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载归因配置: {e}")

    # 加载组合管理器配置
    try:
        with open(f"{config_dir}/portfolio_manager.yaml", "r", encoding="utf-8") as f:
            config["portfolio_manager"] = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"无法加载组合管理器配置: {e}")

    return config


def load_data(data_dir: str = "data") -> pd.DataFrame:
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
        df, exclude_st=True, exclude_suspended=True, min_turnover=0.01, min_market_cap=10  # 10亿市值
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
    MarketFeatures(index_code="000300.SH")
    # 注意：市场特征需要单独计算，这里简化处理

    logger.info("特征计算完成")

    return df


def _load_income_ann_date_map(tushare_root: Path) -> pd.DataFrame:
    income_path = tushare_root / "fundamental" / "income" / "income_all.parquet"
    if not income_path.exists():
        return pd.DataFrame(columns=["ts_code", "end_date", "ann_date"])
    try:
        import pyarrow.parquet as pq

        columns = ["ts_code", "end_date", "ann_date"]
        table = pq.read_table(income_path, columns=columns)
        ann_map = table.to_pandas()
    except Exception as exc:
        logger.warning("读取 income_all 失败: %s", exc)
        return pd.DataFrame(columns=["ts_code", "end_date", "ann_date"])

    ann_map = ann_map.dropna(subset=["ts_code", "end_date"]).copy()
    ann_map["ann_date"] = pd.to_datetime(ann_map["ann_date"], errors="coerce")
    ann_map["end_date"] = pd.to_datetime(ann_map["end_date"], errors="coerce")
    ann_map = ann_map.sort_values(["ts_code", "end_date", "ann_date"])
    ann_map = ann_map.drop_duplicates(subset=["ts_code", "end_date"], keep="last")
    ann_map["ts_code"] = ann_map["ts_code"].astype(str)
    return ann_map[["ts_code", "end_date", "ann_date"]]


def _merge_long_term_fundamentals(df_features: pd.DataFrame) -> pd.DataFrame:
    if df_features is None or df_features.empty:
        return df_features

    if "ts_code" not in df_features.columns:
        if "code" in df_features.columns:
            df_features["ts_code"] = df_features["code"]
        elif "stock" in df_features.columns:
            df_features["ts_code"] = df_features["stock"]

    date_col = (
        "trade_date" if "trade_date" in df_features.columns else "date" if "date" in df_features.columns else None
    )
    if date_col is None:
        return df_features

    trade_dates = pd.to_datetime(df_features[date_col], errors="coerce")
    if trade_dates.isna().all():
        return df_features

    start_date = trade_dates.min().strftime("%Y%m%d")
    end_date = trade_dates.max().strftime("%Y%m%d")
    cache_path = get_data_path("processed") / "long_term_fundamental_features.parquet"

    long_term_df = None
    if cache_path.exists():
        try:
            long_term_df = pd.read_parquet(cache_path)
        except Exception as exc:
            logger.warning("读取长周期特征缓存失败: %s", exc)
            long_term_df = None

    if long_term_df is None or long_term_df.empty:
        try:
            calculator = LongTermFundamentalFeatures(get_tushare_root())
            long_term_df = calculator.calculate_all_features(start_date, end_date)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            long_term_df.to_parquet(cache_path, index=False)
            logger.info("长周期特征已缓存: %s", cache_path)
        except Exception as exc:
            logger.warning("长周期特征计算失败: %s", exc)
            return df_features

    if long_term_df.empty or "end_date" not in long_term_df.columns:
        return df_features

    long_term_df = long_term_df.copy()
    long_term_df["end_date"] = pd.to_datetime(long_term_df["end_date"], errors="coerce")
    ann_map = _load_income_ann_date_map(get_tushare_root())
    if not ann_map.empty:
        long_term_df = long_term_df.merge(ann_map, on=["ts_code", "end_date"], how="left")
        long_term_df["effective_date"] = long_term_df["ann_date"].fillna(long_term_df["end_date"])
    else:
        long_term_df["effective_date"] = long_term_df["end_date"]

    long_term_df["effective_date"] = pd.to_datetime(long_term_df["effective_date"], errors="coerce") + pd.Timedelta(
        days=2
    )

    merged = df_features.copy()
    merged["_trade_date_dt"] = trade_dates
    before_rows = len(merged)
    merged = merged.dropna(subset=["ts_code", "_trade_date_dt"]).copy()
    if len(merged) < before_rows:
        logger.warning("长周期特征合并：丢弃无效日期行=%d", before_rows - len(merged))
    merged = merged.sort_values(["_trade_date_dt", "ts_code"])
    long_term_df = long_term_df.dropna(subset=["ts_code", "effective_date"]).sort_values(["effective_date", "ts_code"])

    merged = pd.merge_asof(
        merged,
        long_term_df,
        by="ts_code",
        left_on="_trade_date_dt",
        right_on="effective_date",
        direction="backward",
    )

    merged = merged.drop(columns=["_trade_date_dt", "ann_date", "effective_date"], errors="ignore")
    logger.info("长周期特征合并完成")
    return merged


def _ensure_industry_l1(df_features: pd.DataFrame, reference_trade_date: str | None) -> pd.DataFrame:
    if df_features is None or df_features.empty:
        return df_features

    if "industry_l1" in df_features.columns:
        return df_features
    if "industry" in df_features.columns:
        df_features = df_features.copy()
        df_features["industry_l1"] = df_features["industry"]
        return df_features
    if "sector" in df_features.columns:
        df_features = df_features.copy()
        df_features["industry_l1"] = df_features["sector"]
        return df_features

    if "ts_code" not in df_features.columns:
        if "code" in df_features.columns:
            df_features["ts_code"] = df_features["code"]
        elif "stock" in df_features.columns:
            df_features["ts_code"] = df_features["stock"]

    fallback = _load_stock_industry_map_fallback(reference_trade_date)
    if fallback.empty:
        return df_features

    df_features = df_features.merge(fallback, on="ts_code", how="left")
    logger.info("行业映射补齐完成: rows=%d", fallback.shape[0])
    return df_features


def _run_macro_prediction(trade_date: str) -> dict:
    """
    运行宏观经济预测，返回预测结果。

    Args:
        trade_date: 交易日期（格式 YYYYMMDD）

    Returns:
        dict: 预测结果，包含 systemic_scenario, opportunity_industries, risk_level 等。
              如果宏观模型不可用，返回默认中性结果。
    """
    default_result = {
        "date": trade_date,
        "systemic_scenario": "NORMAL",
        "opportunity_industries": [],
        "risk_level": "MEDIUM",
        "summary": "宏观模型未启用",
        "available": False,
    }

    if MacroPredictor is None:
        logger.warning("宏观模型未安装，跳过宏观预测")
        return default_result

    try:
        from scripts.data._shared.runtime import get_data_path

        processed_dir = get_data_path("processed")
        macro_path = processed_dir / "macro_features.parquet"
        industry_path = processed_dir / "industry_features.parquet"
        northbound_path = processed_dir / "northbound_features.parquet"

        if not macro_path.exists() or not industry_path.exists():
            logger.warning("宏观数据文件不存在（%s / %s），跳过宏观预测", macro_path, industry_path)
            return default_result

        macro_data = pd.read_parquet(macro_path)
        industry_data = pd.read_parquet(industry_path)
        northbound_data = pd.read_parquet(northbound_path) if northbound_path.exists() else None

        predictor = MacroPredictor(config_path="config/sw_nbs_mapping.yaml")
        date_str = pd.to_datetime(trade_date, format="%Y%m%d").strftime("%Y-%m-%d")
        result = predictor.predict(
            date=date_str,
            macro_data=macro_data,
            industry_data=industry_data,
            northbound_data=northbound_data,
        )
        result["available"] = True
        logger.info(
            "宏观预测完成: scenario=%s, risk=%s, 机会行业=%d个",
            result["systemic_scenario"],
            result["risk_level"],
            len(result.get("opportunity_industries", [])),
        )
        return result

    except Exception:
        logger.exception("宏观预测异常，使用默认中性结果")
        return default_result


def _macro_to_industry_tilts(macro_result: dict) -> dict:
    """
    将宏观预测的机会行业转换为行业倾斜权重。

    景气度评分 → 倾斜权重映射：
    - boom_score >= 70: +0.15
    - boom_score >= 50: +0.10
    - boom_score >= 30: +0.05

    Args:
        macro_result: 宏观预测结果

    Returns:
        dict: {行业名: 倾斜权重}
    """
    tilts = {}
    for ind in macro_result.get("opportunity_industries", []):
        score = ind.get("boom_score", 0)
        if score >= 70:
            tilts[ind["industry"]] = 0.15
        elif score >= 50:
            tilts[ind["industry"]] = 0.10
        elif score >= 30:
            tilts[ind["industry"]] = 0.05
    return tilts


def _macro_risk_to_exposure_factor(macro_result: dict) -> float:
    """
    根据宏观风险等级返回仓位调整因子。

    Returns:
        float: 仓位乘数（0.0 ~ 1.0）
    """
    risk_level = macro_result.get("risk_level", "MEDIUM")
    scenario = macro_result.get("systemic_scenario", "NORMAL")

    if scenario == "SYSTEMIC RECESSION":
        return 0.3  # 系统衰退：仅保留 30% 仓位

    factor_map = {"LOW": 1.0, "MEDIUM": 0.85, "HIGH": 0.6}
    return factor_map.get(risk_level, 0.85)


def _run_long_term_selectors(
    config: dict,
    df_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """运行长周期选股器（价值+成长）

    Args:
        config: portfolio_manager 配置
        df_features: 特征数据

    Returns:
        (value_candidates, growth_candidates) 两个候选池
    """
    pm_cfg = (config.get("portfolio_manager") or {}).get("portfolio_manager", {})
    lt_cfg = pm_cfg.get("long_term_selector", {})

    if not lt_cfg.get("enabled", False):
        logger.info("长周期选股器未启用，跳过")
        return pd.DataFrame(), pd.DataFrame()

    data_root = get_tushare_root()
    value_model_path = Path(lt_cfg.get("value_model_path", "data/models/hybrid_value.pkl"))
    growth_model_path = Path(lt_cfg.get("growth_model_path", "data/models/hybrid_growth.pkl"))
    value_top_n = int(lt_cfg.get("value_top_n", 20))
    growth_top_n = int(lt_cfg.get("growth_top_n", 20))
    rule_only = bool(lt_cfg.get("rule_only", False))
    rule_fallback = bool(lt_cfg.get("rule_fallback", True))
    min_candidates = int(lt_cfg.get("min_candidates", 5))
    rule_top_n = int(lt_cfg.get("rule_top_n", max(value_top_n, growth_top_n)))
    rule_config = lt_cfg.get("rule_config", {}) or {}
    value_rule_cfg = rule_config.get("value", {}) or {}
    growth_rule_cfg = rule_config.get("growth", {}) or {}

    def _to_hybrid_hard_rules(cfg: dict) -> dict:
        hard_filters = cfg.get("hard_filters", {}) or {}
        converted = {}
        for field, rule in hard_filters.items():
            if "eq" in rule:
                converted[field] = (rule.get("eq"),)
            else:
                converted[field] = (rule.get("min"), rule.get("max"))
        return converted

    value_candidates = pd.DataFrame()
    growth_candidates = pd.DataFrame()

    def _run_rule_selector(selector, top_n: int) -> pd.DataFrame:
        try:
            filtered = selector.hard_filter(df_features)
            if filtered.empty:
                return pd.DataFrame()
            scored = selector.calculate_score(filtered)
            return scored.head(top_n)
        except Exception as exc:
            logger.warning("规则选股失败: %s", exc)
            return pd.DataFrame()

    value_rule_candidates = pd.DataFrame()
    growth_rule_candidates = pd.DataFrame()

    if rule_only or rule_fallback:
        value_rule_candidates = _run_rule_selector(
            ValueStockSelector(data_root=data_root, rule_config=value_rule_cfg), rule_top_n
        )
        growth_rule_candidates = _run_rule_selector(
            GrowthStockSelector(data_root=data_root, rule_config=growth_rule_cfg), rule_top_n
        )
        if not value_rule_candidates.empty:
            logger.info("价值股规则候选: %d 只", len(value_rule_candidates))
        if not growth_rule_candidates.empty:
            logger.info("成长股规则候选: %d 只", len(growth_rule_candidates))

    if not rule_only:
        # 价值股选股器（模型选优）
        if value_model_path.exists():
            try:
                value_selector = HybridStockSelector(
                    "value",
                    data_root=data_root,
                    model_path=value_model_path,
                    hard_rules=_to_hybrid_hard_rules(value_rule_cfg),
                )
                value_selector.load_model()
                value_candidates = value_selector.select(df_features, top_n=value_top_n)
                logger.info("价值股选股完成: %d 只候选", len(value_candidates))
            except Exception as exc:
                logger.warning("价值股选股失败: %s", exc)
        else:
            logger.info("价值股模型不存在: %s，跳过", value_model_path)

        # 成长股选股器（模型选优）
        if growth_model_path.exists():
            try:
                growth_selector = HybridStockSelector(
                    "growth",
                    data_root=data_root,
                    model_path=growth_model_path,
                    hard_rules=_to_hybrid_hard_rules(growth_rule_cfg),
                )
                growth_selector.load_model()
                growth_candidates = growth_selector.select(df_features, top_n=growth_top_n)
                logger.info("成长股选股完成: %d 只候选", len(growth_candidates))
            except Exception as exc:
                logger.warning("成长股选股失败: %s", exc)
        else:
            logger.info("成长股模型不存在: %s，跳过", growth_model_path)

    if rule_only:
        value_candidates = value_rule_candidates
        growth_candidates = growth_rule_candidates
    elif rule_fallback:
        if value_candidates.empty or len(value_candidates) < min_candidates:
            logger.info("价值股候选不足，回退规则候选池")
            value_candidates = value_rule_candidates
        if growth_candidates.empty or len(growth_candidates) < min_candidates:
            logger.info("成长股候选不足，回退规则候选池")
            growth_candidates = growth_rule_candidates

    return value_candidates, growth_candidates


def _build_governance_engine(config: dict) -> ChampionChallengerEngine:
    governance_cfg = config.get("strategy_governance") or {}
    governance_raw = governance_cfg.get("strategy_governance", {})
    challenger_weights = governance_cfg.get("challenger_weights", {})
    seed_raw = governance_cfg.get("seed_balance_strategy", {})

    active_champion_id = normalize_strategy_id(governance_raw.get("active_champion_id", "seed_balance_strategy"))
    governance = StrategyGovernanceConfig(
        active_champion_id=active_champion_id,
        champion_source=governance_raw.get("champion_source", "manual"),
        manual_effective_date=governance_raw.get("manual_effective_date"),
        manual_reason=governance_raw.get("manual_reason"),
        challengers=tuple(
            governance_raw.get(
                "challengers",
                [
                    "balance_strategy_v1",
                    "positive_strategy_v1",
                    "value_strategy_v1",
                    "satellite_strategy_v1",
                ],
            )
        ),
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


def _load_stock_industry_map_fallback(reference_trade_date: str | None = None) -> pd.DataFrame:
    """
    当特征侧缺失行业字段时，回退使用申万L1成分映射（sw_index_member + sw_industry_l1）。
    """
    tushare_root = get_tushare_root()
    member_path = tushare_root / "sw_industry" / "sw_index_member.parquet"
    l1_path = tushare_root / "sw_industry" / "sw_industry_l1.parquet"
    if not member_path.exists() or not l1_path.exists():
        return pd.DataFrame(columns=["ts_code", "industry_l1"])

    try:
        members = pd.read_parquet(member_path)
        l1 = pd.read_parquet(l1_path)
    except Exception as exc:
        logger.warning("读取申万映射回退文件失败: %s", exc)
        return pd.DataFrame(columns=["ts_code", "industry_l1"])

    required_member = {"index_code", "con_code"}
    required_l1 = {"index_code"}
    if not required_member.issubset(members.columns) or not required_l1.issubset(l1.columns):
        return pd.DataFrame(columns=["ts_code", "industry_l1"])

    l1_name_col = (
        "industry_name" if "industry_name" in l1.columns else ("index_name" if "index_name" in l1.columns else None)
    )
    if l1_name_col is None:
        return pd.DataFrame(columns=["ts_code", "industry_l1"])

    members = members.copy()
    members["index_code"] = members["index_code"].astype(str)
    members["ts_code"] = members["con_code"].astype(str)

    ref_date = pd.to_datetime(reference_trade_date, errors="coerce")
    if pd.notna(ref_date):
        in_date = pd.to_datetime(members.get("in_date"), errors="coerce")
        out_date = pd.to_datetime(members.get("out_date"), errors="coerce")
        active_mask = (in_date.isna() | (in_date <= ref_date)) & (out_date.isna() | (out_date > ref_date))
        members = members[active_mask].copy()

    l1 = l1[["index_code", l1_name_col]].rename(columns={l1_name_col: "industry_l1"}).copy()
    l1["index_code"] = l1["index_code"].astype(str)
    mapped = members.merge(l1, on="index_code", how="left")
    mapped = mapped.dropna(subset=["ts_code", "industry_l1"]).copy()
    if mapped.empty:
        return pd.DataFrame(columns=["ts_code", "industry_l1"])

    if "in_date" in mapped.columns:
        mapped["in_date"] = pd.to_datetime(mapped["in_date"], errors="coerce")
        mapped = mapped.sort_values(["ts_code", "in_date"]).groupby("ts_code", as_index=False).tail(1)
    else:
        mapped = mapped.drop_duplicates(subset=["ts_code"], keep="last")

    mapped["industry_l1"] = mapped["industry_l1"].astype(str)
    return mapped[["ts_code", "industry_l1"]].reset_index(drop=True)


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
    df_features = _merge_long_term_fundamentals(df_features)

    latest_trade_date_hint = None
    if "trade_date" in df_features.columns:
        max_date = pd.to_datetime(df_features["trade_date"], errors="coerce").max()
        if pd.notna(max_date):
            latest_trade_date_hint = max_date.strftime("%Y%m%d")
    elif "date" in df_features.columns:
        max_date = pd.to_datetime(df_features["date"], errors="coerce").max()
        if pd.notna(max_date):
            latest_trade_date_hint = max_date.strftime("%Y%m%d")

    df_features = _ensure_industry_l1(df_features, latest_trade_date_hint)

    # 3. 创建模型
    trend_model = create_trend_model(config.get("trend_model", {}).get("trend_model", {}).get("model_type", "rule"))

    # 4. 预测趋势状态
    logger.info("预测趋势状态...")
    # 统一股票代码字段
    if "code" not in df_features.columns and "stock" in df_features.columns:
        df_features["code"] = df_features["stock"]
    if "stock" not in df_features.columns and "code" in df_features.columns:
        df_features["stock"] = df_features["code"]

    index_candidates = {"000300.SH", "sh.000300", "000300.SZ"}
    index_code = next((c for c in index_candidates if c in set(df_features["code"].values)), None)
    df_index = df_features[df_features["code"] == index_code].copy() if index_code else pd.DataFrame()

    if df_index.empty:
        df_index = _load_hs300_index_frame(reference_trade_date=latest_trade_date_hint)
        if not df_index.empty:
            logger.info("已从Tushare指数文件加载沪深300: rows=%d", len(df_index))

    if len(df_index) > 0:
        trend_result = trend_model.predict(df_index)
        trend_state = trend_result["state"]
        logger.info(f"趋势状态: {trend_result['state_name']} (state={trend_state})")
    else:
        trend_result = {"position_suggestion": 0.3, "state_name": "震荡"}
        trend_state = 1  # 默认震荡
        logger.warning("无法获取沪深300指数数据，使用默认趋势状态: 震荡")

    # 4.5 宏观经济预测
    logger.info("运行宏观经济预测...")
    macro_trade_date = latest_trade_date_hint or datetime.now().strftime("%Y%m%d")
    macro_result = _run_macro_prediction(macro_trade_date)
    macro_industry_tilts = _macro_to_industry_tilts(macro_result)
    macro_exposure_factor = _macro_risk_to_exposure_factor(macro_result)
    if macro_result.get("available", False):
        logger.info(
            "宏观仓位因子: %.2f, 宏观行业倾斜: %d个行业",
            macro_exposure_factor,
            len(macro_industry_tilts),
        )

    # 4.6 导出宏观信号
    if export_macro_signals is not None:
        try:
            tushare_root = get_tushare_root()
            signal_dir = get_data_path("signals", ensure=True)
            macro_signal_paths = export_macro_signals(
                output_dir=signal_dir,
                delay_days=2,
                spread_threshold=0.5,
                spread_mode="threshold",
                tushare_root=tushare_root,
            )
            exported = {k: str(v) for k, v in macro_signal_paths.items() if v is not None}
            if exported:
                logger.info("宏观信号已导出: %s", exported)
        except Exception:
            logger.warning("宏观信号导出失败，不影响主流程", exc_info=True)

    # 5. 运行Champion/Challenger四策略，执行层只读取Champion
    logger.info("运行选股策略治理（Champion/Challenger）...")
    governance_engine = _build_governance_engine(config)

    if "ts_code" not in df_features.columns and "code" in df_features.columns:
        df_features["ts_code"] = df_features["code"]
    if "trade_date" not in df_features.columns and "date" in df_features.columns:
        df_features["trade_date"] = df_features["date"]

    if "trade_date" in df_features.columns:
        latest_trade_date = pd.to_datetime(df_features["trade_date"].max(), errors="coerce").strftime("%Y%m%d")
    else:
        latest_trade_date = datetime.now().strftime("%Y%m%d")

    governance_cfg = config.get("strategy_governance") or {}
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

    active_champion_id = normalize_strategy_id(promotion_decision.get("next_champion", configured_champion))
    logger.info(
        "自动晋升决策: enabled=%s, promoted=%s, champion=%s -> %s, reason=%s",
        promotion_decision.get("enabled", False),
        promotion_decision.get("promoted", False),
        configured_champion,
        active_champion_id,
        promotion_decision.get("reason"),
    )
    supported_champions = set(governance_engine.governance_config.normalized_challengers())
    supported_champions.add("seed_balance_strategy")
    if active_champion_id not in supported_champions:
        logger.warning(
            "冠军策略未在可用列表中，已回退 seed_balance_strategy: %s",
            active_champion_id,
        )
        promotion_decision = {
            **promotion_decision,
            "next_champion": "seed_balance_strategy",
            "promoted": False,
            "reason": f"fallback:unsupported:{active_champion_id}",
        }
        active_champion_id = "seed_balance_strategy"
    decision_path = Path(auto_cfg.get("decision_path", "data/backtest/governance/promotion_decisions.jsonl"))
    decision_record = {
        "trade_date": latest_trade_date,
        "champion_source": governance_raw.get("champion_source", "manual"),
        **promotion_decision,
    }
    _append_decision_jsonl(decision_path, decision_record)

    seed_data = df_features.copy()
    if "ts_code" in seed_data.columns:
        seed_data = seed_data[~seed_data["ts_code"].isin(index_candidates)].copy()

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

    signals_root = get_data_path("signals", "stock_selector", ensure=True)
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
    overlay_resolved = resolve_industry_overlay_config(
        industry_cfg if isinstance(industry_cfg, dict) else None, trend_state
    )
    overlay_strength = float(overlay_resolved["overlay_strength"])
    mainline_strength = float(overlay_resolved["mainline_strength"])
    signal_weights = overlay_resolved["signal_weights"]
    tilt_strength = float(overlay_resolved.get("tilt_strength", 0.0))
    industry_tilts = overlay_resolved.get("industry_tilts", {})
    # 合并宏观模型的行业倾斜（宏观倾斜不覆盖已有配置，取较大值）
    if macro_industry_tilts and isinstance(industry_tilts, dict):
        for ind_name, macro_tilt in macro_industry_tilts.items():
            existing = industry_tilts.get(ind_name, 0.0)
            industry_tilts[ind_name] = max(existing, macro_tilt)
        logger.info("宏观行业倾斜已合并: 新增/更新 %d 个行业", len(macro_industry_tilts))
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

    industry_snapshot, industry_score_snapshot, industry_snapshot_path, industry_score_snapshot_path = (
        _build_and_load_industry_snapshot(
            latest_trade_date=latest_trade_date,
            lookback_days=lookback_days if isinstance(lookback_days, dict) else None,
            default_lookback_days=int(default_lookback),
            half_life_days=half_life_days if isinstance(half_life_days, dict) else None,
            default_half_life_days=float(default_half_life),
        )
    )
    stock_industry_map = build_stock_industry_map_from_features(df_features)
    feature_industry_map_rows = len(stock_industry_map)
    fallback_industry_map = _load_stock_industry_map_fallback(latest_trade_date)
    if stock_industry_map.empty and not fallback_industry_map.empty:
        stock_industry_map = fallback_industry_map
        logger.info("行业映射回退生效（申万L1成分）: rows=%d", len(stock_industry_map))
    elif not fallback_industry_map.empty:
        stock_industry_map = pd.concat([stock_industry_map, fallback_industry_map], ignore_index=True)
        stock_industry_map = stock_industry_map.dropna(subset=["ts_code", "industry_l1"]).drop_duplicates(
            subset=["ts_code"], keep="first"
        )
        logger.info(
            "行业映射补全完成: feature_rows=%d, fallback_rows=%d, merged_rows=%d",
            feature_industry_map_rows,
            len(fallback_industry_map),
            len(stock_industry_map),
        )
    if stock_industry_map.empty:
        logger.warning("行业映射为空，行业叠加将退化为无行业覆盖")
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

    # 7. Champion执行信号转组合候选（短线动量）
    df_ranked = _signals_to_ranked(execution_signals)
    if df_ranked.empty:
        logger.warning("Champion信号为空，回退到简单排序")
        df_ranked = df_features.copy()
        df_ranked["rank_score"] = np.random.rand(len(df_ranked))
        df_ranked["rank"] = df_ranked["rank_score"].rank(ascending=False)
        if "code" not in df_ranked.columns and "ts_code" in df_ranked.columns:
            df_ranked["code"] = df_ranked["ts_code"]

    # 8. 长周期选股器 + 组合管理器
    value_candidates, growth_candidates = _run_long_term_selectors(config, df_features)
    pm_cfg = (config.get("portfolio_manager") or {}).get("portfolio_manager", {})
    use_portfolio_manager = not value_candidates.empty or not growth_candidates.empty

    if use_portfolio_manager:
        logger.info("使用PortfolioManager整合长短线组合...")
        portfolio_mgr = PortfolioManager(
            long_term_ratio=float(pm_cfg.get("long_term_ratio", 0.70)),
            short_term_ratio=float(pm_cfg.get("short_term_ratio", 0.30)),
            value_growth_ratio=float(pm_cfg.get("value_growth_ratio", 0.50)),
            max_industry_ratio=float(pm_cfg.get("max_industry_ratio", 0.30)),
            min_avg_amount=float(pm_cfg.get("min_avg_amount", 1e8)),
        )
        total_positions = int(pm_cfg.get("total_positions", 10))
        # df_ranked 作为短线动量候选
        portfolio = portfolio_mgr.construct_portfolio(
            value_candidates=value_candidates,
            growth_candidates=growth_candidates,
            momentum_candidates=df_ranked,
            total_positions=total_positions,
        )
        logger.info(
            "PortfolioManager组合: 总%d只 (价值%d, 成长%d, 动量%d)",
            len(portfolio),
            len(portfolio[portfolio["strategy_type"] == "value"]) if "strategy_type" in portfolio.columns else 0,
            len(portfolio[portfolio["strategy_type"] == "growth"]) if "strategy_type" in portfolio.columns else 0,
            len(portfolio[portfolio["strategy_type"] == "momentum"]) if "strategy_type" in portfolio.columns else 0,
        )
    else:
        # 降级模式：纯短线，使用原有PortfolioConstruction
        logger.info("长周期选股器未启用或无候选，使用纯短线组合构建...")
        portfolio_constructor = PortfolioConstruction()
        portfolio = portfolio_constructor.construct_portfolio(df_ranked, trend_state)

    # 9. 风险控制
    logger.info("风险控制...")
    risk_control = RiskControl((config.get("risk_control") or {}).get("risk_control"))
    portfolio = risk_control.adjust_weights(portfolio)

    market_volatility = risk_control.compute_market_volatility(df_index) if df_index is not None else np.nan
    latest_index_return = None
    latest_week_return = None
    if df_index is not None and len(df_index) > 0:
        df_index_sorted = df_index.sort_values("trade_date")
        if "pct_chg" in df_index_sorted.columns:
            index_returns = pd.to_numeric(df_index_sorted["pct_chg"], errors="coerce") / 100.0
        elif "return" in df_index_sorted.columns:
            index_returns = pd.to_numeric(df_index_sorted["return"], errors="coerce")
        elif "close" in df_index_sorted.columns:
            index_returns = pd.to_numeric(df_index_sorted["close"], errors="coerce").pct_change()
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
    if isinstance(trend_result, dict):
        trend_position = float(trend_result.get("position_suggestion", 1.0))
    else:
        trend_position = float(getattr(trend_result, "position_suggestion", 1.0))
    target_exposure = min(position_info["final_position"], trend_position)
    # 宏观风险因子调整目标仓位
    if macro_result.get("available", False) and macro_exposure_factor < 1.0:
        pre_macro_exposure = target_exposure
        target_exposure = target_exposure * macro_exposure_factor
        logger.info(
            "宏观风险调整仓位: %.2f → %.2f (factor=%.2f, scenario=%s)",
            pre_macro_exposure,
            target_exposure,
            macro_exposure_factor,
            macro_result.get("systemic_scenario", "NORMAL"),
        )
    portfolio = risk_control.scale_to_target_exposure(portfolio, target_exposure)
    portfolio_for_risk = portfolio.copy()
    if "sector" not in portfolio_for_risk.columns and "industry_l1" in portfolio_for_risk.columns:
        portfolio_for_risk["sector"] = portfolio_for_risk["industry_l1"]
    risk_checks_raw = risk_control.check_portfolio_risk(portfolio_for_risk) if len(portfolio_for_risk) > 0 else {}
    risk_checks = {k: bool(v) for k, v in risk_checks_raw.items()}
    logger.info(
        "仓位管理: market_vol=%.2f%%, trend_pos=%.2f%%, vol_target_pos=%.2f%%, final_pos=%.2f%%",
        (
            (market_volatility * 100.0)
            if market_volatility is not None and not np.isnan(market_volatility)
            else float("nan")
        ),
        trend_position * 100.0,
        position_info["final_position"] * 100.0,
        float(portfolio["weight"].sum()) * 100.0 if len(portfolio) > 0 else 0.0,
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
    if use_portfolio_manager and "strategy_type" in portfolio.columns:
        for stype in ["value", "growth", "momentum"]:
            sub = portfolio[portfolio["strategy_type"] == stype]
            if not sub.empty:
                logger.info(f"  {stype}: {len(sub)}只, 权重合计{sub['weight'].sum():.2%}")
    code_col = "code" if "code" in portfolio.columns else "ts_code"
    logger.info("\n前5只股票:")
    for _, row in portfolio.head(5).iterrows():
        rank_str = f", 排名 {row['rank']}" if "rank" in row.index and pd.notna(row.get("rank")) else ""
        stype_str = f", 策略 {row['strategy_type']}" if "strategy_type" in row.index else ""
        logger.info(f"  {row[code_col]}: 权重 {row['weight']:.2%}{rank_str}{stype_str}")

    # 11. 保存结果
    output_dir = get_data_path("signals", "portfolio", ensure=True)
    output_file = output_dir / f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv"
    portfolio.to_csv(output_file, index=False)
    logger.info(f"组合结果已保存到: {output_file}")

    # 12. 归因输入快照（因子暴露 + 行业权重）
    attr_cfg = config.get("attribution") or {}
    factor_cfg = (attr_cfg.get("factor_exposure") or {}) if isinstance(attr_cfg, dict) else {}
    if factor_cfg.get("enabled", False):
        factor_cols = factor_cfg.get("factor_cols") or []
        if factor_cols and "trade_date" in df_features.columns:
            date_series = pd.to_datetime(df_features["trade_date"], errors="coerce")
            latest_date = date_series.max()
            latest_date = latest_date.strftime("%Y%m%d") if pd.notna(latest_date) else datetime.now().strftime("%Y%m%d")
            date_str = date_series.dt.strftime("%Y%m%d")
            df_latest = df_features[date_str == latest_date].copy()
            if "ts_code" not in df_latest.columns and "code" in df_latest.columns:
                df_latest["ts_code"] = df_latest["code"]

            neutralize_cfg = factor_cfg.get("neutralize") or {}
            neutralize_industry = bool(neutralize_cfg.get("industry", True))
            zscore = bool(factor_cfg.get("zscore", True))

            try:
                exposures = compute_factor_exposures(
                    df_latest,
                    factor_cols,
                    date_col="trade_date",
                    code_col="ts_code",
                    industry_col="industry_l1",
                    zscore=zscore,
                    neutralize_industry=neutralize_industry,
                )

                code_col = "code" if "code" in portfolio.columns else "ts_code"
                weights = portfolio[[code_col, "weight"]].rename(columns={code_col: "ts_code"})
                exposures = exposures.merge(weights, on="ts_code", how="inner")
                exposures["return"] = pd.NA

                attr_dir = get_data_path("signals", "attribution", ensure=True)
                exposure_path = attr_dir / f"factor_exposure_{latest_date}.csv"
                exposures.to_csv(exposure_path, index=False)
                logger.info("因子暴露快照已保存: %s", exposure_path)

                if "industry_l1" in portfolio.columns:
                    industry_weights = portfolio.groupby("industry_l1", dropna=False)["weight"].sum().reset_index()
                    industry_weights.insert(0, "trade_date", latest_date)
                    industry_weights["return"] = pd.NA
                    industry_path = attr_dir / f"portfolio_industry_{latest_date}.csv"
                    industry_weights.to_csv(industry_path, index=False)
                    logger.info("行业权重快照已保存: %s", industry_path)
            except Exception as exc:
                logger.warning("归因快照生成失败: %s", exc)

    context_file = output_dir / f"execution_context_{datetime.now().strftime('%Y%m%d')}.json"
    context_payload = {
        "trade_date": latest_trade_date,
        "active_champion_id": active_champion_id,
        "portfolio_mode": "long_short" if use_portfolio_manager else "short_only",
        "long_term_selector": {
            "enabled": use_portfolio_manager,
            "value_candidates": len(value_candidates),
            "growth_candidates": len(growth_candidates),
        },
        "signal_contract_path": str(contract_path),
        "execution_signal_path": str(exec_signal_path),
        "unified_signal_contract_path": str(unified_contract_path),
        "industry_snapshot_path": str(industry_snapshot_path) if industry_snapshot_path.exists() else None,
        "industry_score_snapshot_path": (
            str(industry_score_snapshot_path) if industry_score_snapshot_path.exists() else None
        ),
        "industry_overlay_config": {
            "regime": regime_name,
            "overlay_strength": overlay_strength,
            "mainline_strength": mainline_strength,
            "signal_weights": signal_weights,
        },
        "position_info": position_info,
        "target_exposure": float(target_exposure),
        "risk_checks": risk_checks,
        "macro_prediction": {
            "available": macro_result.get("available", False),
            "systemic_scenario": macro_result.get("systemic_scenario", "NORMAL"),
            "risk_level": macro_result.get("risk_level", "MEDIUM"),
            "exposure_factor": macro_exposure_factor,
            "opportunity_industries_count": len(macro_result.get("opportunity_industries", [])),
            "industry_tilts_injected": len(macro_industry_tilts),
        },
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
    trend_model = create_trend_model(config.get("trend_model", {}).get("trend_model", {}).get("model_type", "rule"))

    rank_model_config = config.get("rank_model", {})
    if rank_model_config.get("rank_model", {}).get("enabled", False):
        if RankModelLGBM is None:
            logger.warning("rank_model 启用但 lightgbm 不可用，回退为空模型")
            rank_model = None
        else:
            rank_model = RankModelLGBM(rank_model_config.get("lgbm_params", {}))
    else:
        rank_model = None

    entry_model_config = config.get("entry_model", {})
    if entry_model_config.get("entry_model", {}).get("enabled", False):
        entry_model = EntryModelLR(entry_model_config.get("entry_model", {}))
    else:
        entry_model = None

    # 创建组合构建器和风险控制器
    portfolio_constructor = PortfolioConstruction()
    risk_control = RiskControl()

    # 运行回测
    backtest = WalkForwardBacktest()
    results = backtest.run_backtest(df, trend_model, rank_model, entry_model, portfolio_constructor, risk_control)

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
    output_dir = get_data_path("processed", ensure=True)
    output_file = output_dir / f"backtest_results_{datetime.now().strftime('%Y%m%d')}.csv"
    pd.DataFrame([results["metrics"]]).to_csv(output_file, index=False)
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
    mode = "weekly"  # 默认每周运行模式

    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == "backtest":
        # 回测模式
        run_backtest_workflow(config, df)
    elif mode == "weekly":
        # 每周运行模式
        run_weekly_workflow(config, df)
    else:
        logger.error(f"未知模式: {mode}")
        logger.info("使用方法: python run_weekly.py [weekly|backtest]")

    logger.info("程序结束")


if __name__ == "__main__":
    main()
