"""
多逻辑选股模型（Value / Growth / Frontier）
根据 docs/Chatgpt选股模型方案设计.md 实现
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from scripts.data._shared.runtime import get_tushare_root
logger = logging.getLogger(__name__)


@dataclass
class MultiAlphaConfig:
    lookback_days: int = 120
    winsor_limits: Tuple[float, float] = (0.01, 0.99)
    top_n: int = 30

    # 子组合权重（固定权重方案）
    allocation_fixed: Dict[str, float] = None

    # Regime 动态权重
    allocation_regime: Dict[str, Dict[str, float]] = None

    # Value 评分权重
    value_weights: Dict[str, float] = None

    # Growth 评分权重
    growth_weights: Dict[str, float] = None

    # Frontier 评分权重
    frontier_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.allocation_fixed is None:
            self.allocation_fixed = {
                "value": 0.4,
                "growth": 0.4,
                "frontier": 0.2,
            }
        if self.allocation_regime is None:
            self.allocation_regime = {
                "bear": {"value": 0.6, "growth": 0.2, "frontier": 0.2},
                "sideways": {"value": 0.4, "growth": 0.3, "frontier": 0.3},
                "bull": {"value": 0.2, "growth": 0.5, "frontier": 0.3},
            }
        if self.value_weights is None:
            self.value_weights = {
                "value": 0.35,
                "quality": 0.30,
                "low_vol": 0.20,
                "stability": 0.15,
            }
        if self.growth_weights is None:
            self.growth_weights = {
                "growth": 0.35,
                "accel": 0.25,
                "rps": 0.25,
                "elasticity": 0.15,
            }
        if self.frontier_weights is None:
            self.frontier_weights = {
                "capex": 0.30,
                "turnaround": 0.25,
                "order": 0.25,
                "not_priced": 0.20,
            }


class MultiAlphaStockSelector:
    """多逻辑选股模型"""

    def __init__(self, data_dir: str = None, config: Optional[MultiAlphaConfig] = None):
        self.data_dir = Path(data_dir or str(get_tushare_root()))
        self.daily_dir = self.data_dir / "daily"
        self.fina_dir = self.data_dir / "fundamental"
        self.daily_basic_path = self.data_dir / "daily_basic_all.parquet"
        self.config = config or MultiAlphaConfig()

    # -----------------------------
    # 公共入口
    # -----------------------------
    def select(self, trade_date: str, top_n: Optional[int] = None,
               allocation_method: str = "fixed", regime: str = "sideways") -> Dict[str, pd.DataFrame]:
        """
        运行三套选股并返回结果

        Args:
            trade_date: 交易日期，格式 YYYYMMDD
            top_n: 每个子组合选股数量
            allocation_method: fixed / regime
            regime: bear / sideways / bull
        """
        top_n = top_n or self.config.top_n

        logger.info("加载价格数据与因子数据: %s", trade_date)
        price_snapshot = self._build_price_snapshot(trade_date)
        if price_snapshot.empty:
            raise ValueError(f"未找到 {trade_date} 的日线数据")

        logger.info("加载 daily_basic: %s", trade_date)
        basic_snapshot = self._load_daily_basic(trade_date)

        logger.info("加载财务指标 (fina_indicator) 并对齐 ann_date")
        fina_latest = self._load_fina_indicator_latest(trade_date, universe=price_snapshot["ts_code"].unique())

        # 合并
        df = price_snapshot.merge(basic_snapshot, on=["ts_code", "trade_date"], how="left")
        df = df.merge(fina_latest, on="ts_code", how="left")

        # 评分
        value_scores = self._score_value(df)
        growth_scores = self._score_growth(df)
        frontier_scores = self._score_frontier(df)

        # 组合分配
        alloc = self._get_allocation_weights(allocation_method, regime)

        # 汇总
        score_table = df[["ts_code", "trade_date", "close"]].copy()
        score_table = score_table.merge(value_scores, on="ts_code", how="left")
        score_table = score_table.merge(growth_scores, on="ts_code", how="left")
        score_table = score_table.merge(frontier_scores, on="ts_code", how="left")

        score_table["combined_score"] = (
            score_table["value_score"] * alloc["value"]
            + score_table["growth_score"] * alloc["growth"]
            + score_table["frontier_score"] * alloc["frontier"]
        )

        # 结果筛选
        result = {
            "value": self._topn(score_table, "value_score", top_n),
            "growth": self._topn(score_table, "growth_score", top_n),
            "frontier": self._topn(score_table, "frontier_score", top_n),
            "combined": self._topn(score_table, "combined_score", top_n),
            "all_scores": score_table,
        }
        return result

    # -----------------------------
    # 数据加载
    # -----------------------------
    def _read_parquet_filtered(self, path: Path, filters: Optional[List[Tuple[str, str, str]]] = None,
                               columns: Optional[List[str]] = None) -> pd.DataFrame:
        """带过滤条件的 parquet 读取（兼容不支持 filters 的环境）"""
        if not path.exists():
            logger.warning("文件不存在: %s", path)
            return pd.DataFrame()

        try:
            return pd.read_parquet(path, filters=filters, columns=columns)
        except TypeError:
            # pandas 旧版本不支持 filters
            df = pd.read_parquet(path, columns=columns)
            if filters:
                for col, op, val in filters:
                    if op == "==":
                        df = df[df[col] == val]
                    elif op == "<=":
                        df = df[df[col] <= val]
                    elif op == ">=":
                        df = df[df[col] >= val]
            return df

    def _load_daily_window(self, trade_date: str, lookback_days: int) -> pd.DataFrame:
        year = int(trade_date[:4])
        frames = []
        for y in [year - 1, year]:
            path = self.daily_dir / f"daily_{y}.parquet"
            if path.exists():
                df = self._read_parquet_filtered(
                    path,
                    filters=[("trade_date", "<=", trade_date)],
                    columns=["ts_code", "trade_date", "close"],
                )
                frames.append(df)
        if not frames:
            return pd.DataFrame()

        daily = pd.concat(frames, ignore_index=True)
        daily = daily[daily["trade_date"] <= trade_date]
        daily = daily.sort_values(["ts_code", "trade_date"])

        # 只保留最近 lookback_days + 1 个交易日
        daily = daily.groupby("ts_code", group_keys=False).tail(lookback_days + 1)
        return daily

    def _load_daily_basic(self, trade_date: str) -> pd.DataFrame:
        if not self.daily_basic_path.exists():
            logger.warning("daily_basic 文件不存在: %s", self.daily_basic_path)
            return pd.DataFrame(columns=["ts_code", "trade_date"])

        columns = [
            "ts_code",
            "trade_date",
            "pe_ttm",
            "pb",
            "dv_ttm",
            "total_mv",
            "turnover_rate_f",
            "volume_ratio",
        ]
        df = self._read_parquet_filtered(
            self.daily_basic_path,
            filters=[("trade_date", "==", trade_date)],
            columns=columns,
        )
        return df

    def _load_fina_indicator_latest(self, trade_date: str, universe: Optional[np.ndarray] = None) -> pd.DataFrame:
        year = int(trade_date[:4])
        years = [year, year - 1]

        columns = [
            "ts_code",
            "ann_date",
            "end_date",
            "roe_dt",
            "ocfps",
            "grossprofit_margin",
            "netprofit_yoy",
            "dt_netprofit_yoy",
            "or_yoy",
            "ocf_yoy",
            "fixed_assets",
            "assets_yoy",
        ]

        frames = []
        for y in years:
            path = self.fina_dir / f"fina_indicator_{y}.parquet"
            if path.exists():
                df = self._read_parquet_filtered(
                    path,
                    filters=[("ann_date", "<=", trade_date)],
                    columns=columns,
                )
                frames.append(df)

        if not frames:
            empty_cols = columns + [
                "profit_yoy",
                "profit_yoy_diff",
                "or_yoy_diff",
                "roe_dt_diff",
                "grossprofit_margin_diff",
                "fixed_assets_diff",
                "assets_yoy_diff",
            ]
            return pd.DataFrame(columns=empty_cols)

        fina = pd.concat(frames, ignore_index=True)
        fina = fina[fina["ann_date"] <= trade_date]
        if universe is not None:
            fina = fina[fina["ts_code"].isin(universe)]

        # 数值列转换
        numeric_cols = [c for c in columns if c not in {"ts_code", "ann_date", "end_date"}]
        for col in numeric_cols:
            fina[col] = pd.to_numeric(fina[col], errors="coerce")

        fina = fina.sort_values(["ts_code", "ann_date", "end_date"])

        # 组合利润YoY（优先 netprofit_yoy）
        fina["profit_yoy"] = fina["netprofit_yoy"].where(~fina["netprofit_yoy"].isna(), fina["dt_netprofit_yoy"])

        # 计算变化（加速度/拐点）
        diff_cols = [
            "profit_yoy",
            "or_yoy",
            "roe_dt",
            "grossprofit_margin",
            "fixed_assets",
            "assets_yoy",
        ]
        for col in diff_cols:
            fina[f"{col}_diff"] = fina.groupby("ts_code")[col].diff()

        # 保留每只股票最新披露记录
        latest = fina.groupby("ts_code", as_index=False).tail(1)

        return latest

    # -----------------------------
    # 因子 & 评分
    # -----------------------------
    def _build_price_snapshot(self, trade_date: str) -> pd.DataFrame:
        daily = self._load_daily_window(trade_date, self.config.lookback_days)
        if daily.empty:
            return daily

        daily = daily.sort_values(["ts_code", "trade_date"])
        grp = daily.groupby("ts_code", group_keys=False)

        daily["ret"] = grp["close"].pct_change()
        daily["ret_60"] = grp["close"].pct_change(60)
        daily["ret_120"] = grp["close"].pct_change(120)

        daily["ma20"] = grp["close"].rolling(20).mean().reset_index(level=0, drop=True)
        daily["above_ma20"] = (daily["close"] > daily["ma20"]).astype(float)
        daily["days_above_ma20_60"] = grp["above_ma20"].rolling(60).mean().reset_index(level=0, drop=True)

        daily["vol_60"] = grp["ret"].rolling(60).std().reset_index(level=0, drop=True)
        daily["down_ret"] = daily["ret"].where(daily["ret"] < 0, 0.0)
        daily["down_vol_60"] = grp["down_ret"].rolling(60).std().reset_index(level=0, drop=True)

        roll_max = grp["close"].rolling(120).max().reset_index(level=0, drop=True)
        daily["dd_120"] = daily["close"] / roll_max - 1
        daily["max_dd_120"] = grp["dd_120"].rolling(120).min().reset_index(level=0, drop=True)

        snapshot = daily[daily["trade_date"] == trade_date].copy()
        snapshot["rps_120"] = snapshot["ret_120"].rank(pct=True)
        return snapshot[[
            "ts_code",
            "trade_date",
            "close",
            "ret_60",
            "ret_120",
            "vol_60",
            "down_vol_60",
            "max_dd_120",
            "days_above_ma20_60",
            "rps_120",
        ]]

    def _winsorize(self, series: pd.Series) -> pd.Series:
        s = series.copy()
        s = s.replace([np.inf, -np.inf], np.nan)
        if s.dropna().empty:
            return s
        lower, upper = s.quantile(self.config.winsor_limits)
        return s.clip(lower, upper)

    def _zscore(self, series: pd.Series) -> pd.Series:
        s = series.astype(float)
        s = s.replace([np.inf, -np.inf], np.nan)
        if s.dropna().empty:
            return s * np.nan
        s = self._winsorize(s)
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return s * np.nan
        return (s - s.mean()) / std

    def _score_value(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        value = (
            -self._zscore(data["pe_ttm"]) - self._zscore(data["pb"]) + self._zscore(data["dv_ttm"])
        )
        quality = (
            self._zscore(data["roe_dt"]) + self._zscore(data["ocfps"]) + self._zscore(data["grossprofit_margin"])
        )
        max_dd_abs = -data["max_dd_120"]
        low_vol = -self._zscore(data["vol_60"]) - self._zscore(max_dd_abs)
        stability = self._zscore(data["days_above_ma20_60"]) - self._zscore(data["down_vol_60"])

        score = (
            self.config.value_weights["value"] * value
            + self.config.value_weights["quality"] * quality
            + self.config.value_weights["low_vol"] * low_vol
            + self.config.value_weights["stability"] * stability
        )

        result = pd.DataFrame({
            "ts_code": data["ts_code"],
            "value_score": score,
            "value_component": value,
            "quality_component": quality,
            "low_vol_component": low_vol,
            "stability_component": stability,
        })
        return result

    def _score_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        revenue_yoy = data["or_yoy"]
        profit_yoy = data["profit_yoy"]
        growth = self._zscore(revenue_yoy) + self._zscore(profit_yoy)

        accel = self._zscore(data["profit_yoy_diff"])
        rps = self._zscore(data["rps_120"])
        elasticity = -self._zscore(data["total_mv"])

        score = (
            self.config.growth_weights["growth"] * growth
            + self.config.growth_weights["accel"] * accel
            + self.config.growth_weights["rps"] * rps
            + self.config.growth_weights["elasticity"] * elasticity
        )

        result = pd.DataFrame({
            "ts_code": data["ts_code"],
            "growth_score": score,
            "growth_component": growth,
            "accel_component": accel,
            "rps_component": rps,
            "elasticity_component": elasticity,
        })
        return result

    def _score_frontier(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        capex_base = data["fixed_assets_diff"].fillna(data["assets_yoy_diff"])
        capex = self._zscore(capex_base)

        turnaround = self._zscore(data["roe_dt_diff"]) + self._zscore(data["grossprofit_margin_diff"])
        order_signal = self._zscore(data["ocf_yoy"])
        not_priced = -self._zscore(data["rps_120"])

        score = (
            self.config.frontier_weights["capex"] * capex
            + self.config.frontier_weights["turnaround"] * turnaround
            + self.config.frontier_weights["order"] * order_signal
            + self.config.frontier_weights["not_priced"] * not_priced
        )

        result = pd.DataFrame({
            "ts_code": data["ts_code"],
            "frontier_score": score,
            "capex_component": capex,
            "turnaround_component": turnaround,
            "order_component": order_signal,
            "not_priced_component": not_priced,
        })
        return result

    # -----------------------------
    # 权重与排序
    # -----------------------------
    def _get_allocation_weights(self, method: str, regime: str) -> Dict[str, float]:
        method = method.lower()
        if method == "fixed":
            return self.config.allocation_fixed
        if method == "regime":
            return self.config.allocation_regime.get(regime, self.config.allocation_fixed)
        logger.warning("未知 allocation_method=%s，回退为 fixed", method)
        return self.config.allocation_fixed

    def _topn(self, df: pd.DataFrame, score_col: str, n: int) -> pd.DataFrame:
        temp = df[["ts_code", "trade_date", "close", score_col]].copy()
        temp = temp.dropna(subset=[score_col])
        temp = temp.sort_values(score_col, ascending=False).head(n)
        temp = temp.reset_index(drop=True)
        temp["rank"] = temp.index + 1
        return temp
