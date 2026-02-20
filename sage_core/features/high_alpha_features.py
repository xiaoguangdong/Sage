#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
高 Alpha 因子特征工程

本模块实现了四类高 Alpha 因子的计算：

1. **资金流因子细分**（基于 moneyflow 数据）
   - large_net_inflow_ratio: 大单净流入占比
   - main_inflow_days: 主力资金持续流入天数
   - retail_inst_divergence: 散户/机构资金分化度
   - super_large_ratio: 特大单占比

2. **北向资金细分**（基于 northbound_hold 数据）
   - config_fund_ratio: 配置型资金占比
   - trading_fund_inflow: 交易型资金流入
   - holding_concentration_change: 持仓集中度变化
   - holding_stability: 持仓稳定性

3. **融资融券因子**（基于 margin 数据）
   - margin_balance_change_rate: 融资余额变化率
   - margin_net_buy_ratio: 融资净买入占比

4. **分析师预期因子**（基于 forecast/express 数据）
   - analyst_upgrade_count: 盈利预测上调次数
   - analyst_revision_magnitude: 预期修正幅度
   - analyst_surprise_degree: 超预期程度
   - analyst_consensus: 预期一致性

使用示例：
    >>> from pathlib import Path
    >>> calculator = HighAlphaFeatures(Path("data/tushare"))
    >>>
    >>> # 计算资金流因子
    >>> moneyflow_feat = calculator.compute_moneyflow_features("20240101", lookback_days=20)
    >>> print(moneyflow_feat.head())
    >>>
    >>> # 计算北向资金因子
    >>> northbound_feat = calculator.compute_northbound_features("20240101", lookback_days=60)
    >>>
    >>> # 计算融资融券因子
    >>> margin_feat = calculator.compute_margin_features("20240101", lookback_days=20)
    >>>
    >>> # 计算分析师预期因子
    >>> analyst_feat = calculator.compute_analyst_expectation_features("20240101", lookback_months=3)

数据依赖：
    - moneyflow: data/tushare/moneyflow/*.parquet
    - northbound_hold: data/tushare/northbound/northbound_hold.parquet
    - margin: data/tushare/margin.parquet
    - forecast: data/tushare/fundamental/forecast/*.parquet
    - express: data/tushare/fundamental/express/*.parquet

注意事项：
    - 所有日期格式为 YYYYMMDD 字符串
    - 如果数据不存在或不足，返回空 DataFrame
    - 因子值已标准化到合理范围，无需额外处理
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class HighAlphaFeatures:
    """高 Alpha 因子计算器"""

    def __init__(self, data_root: Path):
        """初始化

        Args:
            data_root: Tushare 数据根目录
        """
        self.data_root = Path(data_root)
        self.moneyflow_dir = self.data_root / "moneyflow"
        self.northbound_dir = self.data_root / "northbound"
        self.margin_path = self.data_root / "margin.parquet"
        self.forecast_dir = self.data_root / "fundamental" / "forecast"
        self.express_dir = self.data_root / "fundamental" / "express"

    # ==================== 资金流因子细分 ====================

    def compute_moneyflow_features(
        self,
        trade_date: str,
        lookback_days: int = 20,
    ) -> pd.DataFrame:
        """计算资金流因子细分

        基于 moneyflow 数据计算4个细分因子，捕捉不同类型资金的行为特征。

        因子说明：
            1. large_net_inflow_ratio: 大单净流入占比
               - 计算：(大单买入 - 大单卖出) / 总成交额
               - 范围：[-1, 1]
               - 含义：正值表示大单净流入，负值表示净流出

            2. main_inflow_days: 主力资金持续流入天数
               - 计算：从最近一天往前数，连续大单净流入的天数
               - 范围：[0, lookback_days]
               - 含义：值越大表示主力资金持续看好

            3. retail_inst_divergence: 散户/机构资金分化度
               - 计算：-corr(小单净流入, 大单净流入)
               - 范围：[-1, 1]
               - 含义：正值表示散户和机构反向操作（分化），负值表示同向

            4. super_large_ratio: 特大单占比
               - 计算：特大单成交额 / 总成交额
               - 范围：[0, 1]
               - 含义：值越大表示机构活跃度越高

        Args:
            trade_date: 交易日期（YYYYMMDD格式）
            lookback_days: 回溯天数，默认20天

        Returns:
            pd.DataFrame: 包含 ts_code 和4个因子列的DataFrame
                如果数据不存在或不足，返回空 DataFrame

        Raises:
            无异常抛出，数据问题返回空 DataFrame

        Example:
            >>> calculator = HighAlphaFeatures(Path("data/tushare"))
            >>> factors = calculator.compute_moneyflow_features("20240101", lookback_days=20)
            >>> print(factors.columns)
            Index(['ts_code', 'large_net_inflow_ratio', 'main_inflow_days',
                   'retail_inst_divergence', 'super_large_ratio'], dtype='object')
            >>> print(factors.head())
        """
        if not self.moneyflow_dir.exists():
            return pd.DataFrame()

        # 读取所有股票的资金流数据
        frames = []
        for file in self.moneyflow_dir.glob("*.parquet"):
            df = pd.read_parquet(file)
            df = df[df["trade_date"] <= trade_date].tail(lookback_days)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        moneyflow = pd.concat(frames, ignore_index=True)
        if "ts_code" not in moneyflow.columns:
            return pd.DataFrame()

        # 预计算净流入
        moneyflow["large_net"] = moneyflow["buy_lg_amount"] - moneyflow["sell_lg_amount"]
        moneyflow["small_net"] = moneyflow["buy_sm_amount"] - moneyflow["sell_sm_amount"]
        moneyflow["super_large_amount"] = moneyflow["buy_elg_amount"] + moneyflow["sell_elg_amount"]
        total_amount_cols = [
            "buy_sm_amount",
            "sell_sm_amount",
            "buy_md_amount",
            "sell_md_amount",
            "buy_lg_amount",
            "sell_lg_amount",
            "buy_elg_amount",
            "sell_elg_amount",
        ]
        moneyflow["total_amount"] = moneyflow[total_amount_cols].sum(axis=1)

        agg = (
            moneyflow.groupby("ts_code")
            .agg(
                large_net_sum=("large_net", "sum"),
                total_amount_sum=("total_amount", "sum"),
                super_large_sum=("super_large_amount", "sum"),
                obs=("large_net", "count"),
                small_std=("small_net", "std"),
                large_std=("large_net", "std"),
            )
            .reset_index()
        )
        agg = agg[agg["obs"] >= 5]
        if agg.empty:
            return pd.DataFrame()

        # 1. 大单净流入占比
        agg["large_net_inflow_ratio"] = agg["large_net_sum"] / agg["total_amount_sum"].replace(0, pd.NA)
        agg["large_net_inflow_ratio"] = agg["large_net_inflow_ratio"].fillna(0.0)

        # 2. 主力资金持续流入天数（从最近往前数）
        def _inflow_days(series: pd.Series) -> int:
            count = 0
            for val in series.iloc[::-1]:
                if val > 0:
                    count += 1
                else:
                    break
            return int(count)

        inflow_days = moneyflow.groupby("ts_code")["large_net"].apply(_inflow_days)
        agg = agg.merge(inflow_days.rename("main_inflow_days"), on="ts_code", how="left")
        agg["main_inflow_days"] = agg["main_inflow_days"].fillna(0).astype(int)

        # 3. 散户/机构资金分化度（负相关表示分化）
        def _divergence(group: pd.DataFrame) -> float:
            if group["small_net"].std() > 0 and group["large_net"].std() > 0:
                return float(-group["small_net"].corr(group["large_net"]))
            return 0.0

        divergence = moneyflow.groupby("ts_code").apply(_divergence)
        agg = agg.merge(divergence.rename("retail_inst_divergence"), on="ts_code", how="left")

        # 4. 特大单占比（机构活跃度）
        agg["super_large_ratio"] = agg["super_large_sum"] / agg["total_amount_sum"].replace(0, pd.NA)
        agg["super_large_ratio"] = agg["super_large_ratio"].fillna(0.0)

        return agg[
            [
                "ts_code",
                "large_net_inflow_ratio",
                "main_inflow_days",
                "retail_inst_divergence",
                "super_large_ratio",
            ]
        ]

    # ==================== 北向资金细分 ====================

    def compute_northbound_features(
        self,
        trade_date: str,
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        """计算北向资金细分因子

        基于 northbound_hold 数据计算4个细分因子，捕捉外资行为特征。

        因子说明：
            1. config_fund_ratio: 配置型资金占比
               - 计算：持仓变化稳定的资金占比
               - 范围：[0, 1]
               - 含义：值越大表示长期配置资金越多

            2. trading_fund_inflow: 交易型资金流入
               - 计算：短期持仓变化的净流入
               - 含义：正值表示交易型资金流入

            3. holding_concentration_change: 持仓集中度变化
               - 计算：持仓集中度的变化率
               - 含义：正值表示持仓更集中

            4. holding_stability: 持仓稳定性
               - 计算：持仓变化的标准差（取负）
               - 含义：值越大表示持仓越稳定

        Args:
            trade_date: 交易日期（YYYYMMDD格式）
            lookback_days: 回溯天数，默认60天（北向数据建议用更长周期）

        Returns:
            pd.DataFrame: 包含 ts_code 和4个因子列的DataFrame
                如果数据不存在或不足，返回空 DataFrame

        Note:
            北向资金数据可能存在缺口，建议先运行 validate_northbound.py 检查数据质量

        Example:
            >>> calculator = HighAlphaFeatures(Path("data/tushare"))
            >>> factors = calculator.compute_northbound_features("20240101", lookback_days=60)
        """
        hk_hold_path = self.northbound_dir / "northbound_hold.parquet"
        if not hk_hold_path.exists():
            return pd.DataFrame()

        # 读取北向持股数据
        hk_hold = pd.read_parquet(hk_hold_path)
        hk_hold = hk_hold[hk_hold["trade_date"] <= trade_date].copy()

        if hk_hold.empty:
            return pd.DataFrame()

        if "ts_code" not in hk_hold.columns or "ratio" not in hk_hold.columns:
            return pd.DataFrame()

        hk_hold = hk_hold.sort_values(["ts_code", "trade_date"])
        hk_hold = hk_hold.groupby("ts_code").tail(lookback_days)

        # 1. 配置型资金占比（持股占比变化 < 5%）
        hk_hold["ratio_change"] = hk_hold.groupby("ts_code")["ratio"].pct_change().abs()
        config_ratio = (
            hk_hold.groupby("ts_code")["ratio_change"]
            .apply(lambda s: float((s < 0.05).sum() / max(len(s), 1)))
            .rename("config_fund_ratio")
        )

        # 2/3/4 聚合
        agg = (
            hk_hold.groupby("ts_code")["ratio"]
            .agg(ratio_std="std", ratio_last="last", ratio_first="first", obs="count")
            .reset_index()
        )
        agg = agg[agg["obs"] >= 10]
        if agg.empty:
            return pd.DataFrame()

        agg["ratio_change_total"] = agg["ratio_last"] - agg["ratio_first"]
        agg["trading_fund_inflow"] = ((agg["ratio_std"] > 0.10) & (agg["ratio_change_total"] > 0)).astype(int)
        agg["holding_concentration_change"] = agg["ratio_change_total"]
        agg["holding_stability"] = -agg["ratio_std"].fillna(0.0)

        agg = agg.merge(config_ratio, on="ts_code", how="left")
        agg["config_fund_ratio"] = agg["config_fund_ratio"].fillna(0.0)

        return agg[
            [
                "ts_code",
                "config_fund_ratio",
                "trading_fund_inflow",
                "holding_concentration_change",
                "holding_stability",
            ]
        ]

    # ==================== 融资融券因子 ====================

    def compute_margin_features(
        self,
        trade_date: str,
        lookback_days: int = 20,
    ) -> pd.DataFrame:
        """计算融资融券因子（市场汇总级别）

        基于 margin 数据计算2个市场级别因子，反映融资资金的整体行为。

        因子说明：
            1. margin_balance_change_rate: 融资余额变化率
               - 计算：(当前融资余额 - 过去融资余额) / 过去融资余额
               - 含义：正值表示融资余额增加，市场情绪乐观

            2. margin_net_buy_ratio: 融资净买入占比
               - 计算：(融资买入 - 融资偿还) / 融资余额
               - 含义：正值表示融资净买入，负值表示净偿还

        Args:
            trade_date: 交易日期（YYYYMMDD格式）
            lookback_days: 回溯天数，默认20天

        Returns:
            pd.DataFrame: 包含 trade_date 和2个因子列的DataFrame
                注意：返回的是市场级别因子，不是个股级别
                如果数据不存在或不足，返回空 DataFrame

        Note:
            - 这是市场级别因子，需要在使用时广播到所有股票
            - 融资融券数据按交易所汇总（上交所+深交所）

        Example:
            >>> calculator = HighAlphaFeatures(Path("data/tushare"))
            >>> factors = calculator.compute_margin_features("20240101", lookback_days=20)
            >>> print(factors)
               trade_date  margin_balance_change_rate  margin_net_buy_ratio
            0    20240101                      0.0523                0.0012
        """
        if not self.margin_path.exists():
            return pd.DataFrame()

        # 读取融资融券数据
        margin = pd.read_parquet(self.margin_path)
        margin = margin[margin["trade_date"] <= trade_date].copy()

        if margin.empty:
            return pd.DataFrame()

        # 按交易所汇总
        margin_agg = margin.groupby("trade_date").agg({"rzye": "sum", "rzmre": "sum", "rzche": "sum"}).reset_index()
        margin_agg = margin_agg.sort_values("trade_date").tail(lookback_days + 1)

        if len(margin_agg) < 2:
            return pd.DataFrame()

        # 1. 融资余额变化率
        current_balance = margin_agg["rzye"].iloc[-1]
        past_balance = margin_agg["rzye"].iloc[0]
        margin_balance_change_rate = (current_balance - past_balance) / past_balance if past_balance > 0 else 0

        # 2. 融资净买入占比（最近一天）
        latest = margin_agg.iloc[-1]
        margin_net_buy = latest["rzmre"] - latest["rzche"]
        margin_net_buy_ratio = margin_net_buy / latest["rzye"] if latest["rzye"] > 0 else 0

        return pd.DataFrame(
            [
                {
                    "trade_date": trade_date,
                    "margin_balance_change_rate": margin_balance_change_rate,
                    "margin_net_buy_ratio": margin_net_buy_ratio,
                }
            ]
        )

    # ==================== 分析师预期因子 ====================

    def compute_analyst_expectation_features(
        self,
        trade_date: str,
        lookback_months: int = 3,
    ) -> pd.DataFrame:
        """计算分析师预期因子

        基于 forecast/express 数据计算4个分析师预期因子，捕捉市场预期变化。

        因子说明：
            1. earnings_upgrade_count: 盈利预测上调次数
               - 计算：回溯期内预增/略增/扭亏/续盈的次数
               - 含义：值越大表示分析师越看好

            2. expectation_revision_rate: 预期修正幅度
               - 计算：最新预告 vs 上次预告的变化率
               - 含义：正值表示预期上调，负值表示下调

            3. beat_expectation_rate: 超预期程度
               - 计算：实际业绩 vs 预期业绩的差异
               - 含义：正值表示超预期，负值表示低于预期

            4. expectation_consistency: 预期一致性
               - 计算：多次预告的标准差（取负）
               - 含义：值越大表示预期越一致

        Args:
            trade_date: 交易日期（YYYYMMDD格式）
            lookback_months: 回溯月数，默认3个月

        Returns:
            pd.DataFrame: 包含 ts_code 和4个因子列的DataFrame
                如果数据不存在或不足，返回空 DataFrame

        Note:
            - 需要 forecast_all.parquet 和 express_all.parquet 文件
            - 分析师数据更新频率较低，建议用更长回溯期（3-6个月）

        Example:
            >>> calculator = HighAlphaFeatures(Path("data/tushare"))
            >>> factors = calculator.compute_analyst_expectation_features("20240101", lookback_months=3)
            >>> print(factors.columns)
            Index(['ts_code', 'earnings_upgrade_count', 'expectation_revision_rate',
                   'beat_expectation_rate', 'expectation_consistency'], dtype='object')
        """
        forecast_path = self.forecast_dir / "forecast_all.parquet"
        express_path = self.express_dir / "express_all.parquet"

        if not forecast_path.exists():
            return pd.DataFrame()

        # 读取业绩预告数据
        forecast = pd.read_parquet(forecast_path)
        forecast = forecast[forecast["ann_date"] <= trade_date].copy()

        if forecast.empty:
            return pd.DataFrame()

        # 计算回溯日期
        trade_dt = pd.to_datetime(trade_date, format="%Y%m%d")
        lookback_date = (trade_dt - pd.DateOffset(months=lookback_months)).strftime("%Y%m%d")
        forecast = forecast[forecast["ann_date"] >= lookback_date]

        express_by_code = None
        if express_path.exists():
            express = pd.read_parquet(express_path)
            express = express[express["ann_date"] <= trade_date].copy()
            if not express.empty:
                express_by_code = {
                    ts_code: group.sort_values("ann_date") for ts_code, group in express.groupby("ts_code")
                }

        # 按股票分组计算
        features = []
        for ts_code, group in forecast.groupby("ts_code"):
            if len(group) < 1:
                continue

            # 1. 盈利上调次数（预增/略增）
            upgrade_types = ["预增", "略增", "扭亏", "续盈"]
            earnings_upgrade_count = group[group["type"].isin(upgrade_types)].shape[0]

            # 2. 预期修正幅度（最新预告 vs 上次预告）
            group_sorted = group.sort_values("ann_date")
            if len(group_sorted) >= 2:
                latest = group_sorted.iloc[-1]
                previous = group_sorted.iloc[-2]
                latest_mid = (latest.get("p_change_min", 0) + latest.get("p_change_max", 0)) / 2
                previous_mid = (previous.get("p_change_min", 0) + previous.get("p_change_max", 0)) / 2
                expectation_revision_rate = (latest_mid - previous_mid) / abs(previous_mid) if previous_mid != 0 else 0
            else:
                expectation_revision_rate = 0

            # 3. 超预期程度（需要 express 数据）
            beat_expectation_rate = 0
            express_stock = express_by_code.get(ts_code) if express_by_code else None
            if express_stock is not None and not express_stock.empty and not group_sorted.empty:
                latest_express = express_stock.iloc[-1]
                latest_forecast = group_sorted.iloc[-1]
                forecast_mid = (latest_forecast.get("net_profit_min", 0) + latest_forecast.get("net_profit_max", 0)) / 2
                actual = latest_express.get("n_income", 0)
                beat_expectation_rate = (actual - forecast_mid) / abs(forecast_mid) if forecast_mid != 0 else 0

            # 4. 预期一致性（预告区间宽度 / 中值，越小越好）
            latest = group_sorted.iloc[-1]
            p_min = latest.get("p_change_min", 0)
            p_max = latest.get("p_change_max", 0)
            p_mid = (p_min + p_max) / 2
            expectation_consistency = (p_max - p_min) / abs(p_mid) if p_mid != 0 else 0

            features.append(
                {
                    "ts_code": ts_code,
                    "earnings_upgrade_count": earnings_upgrade_count,
                    "expectation_revision_rate": expectation_revision_rate,
                    "beat_expectation_rate": beat_expectation_rate,
                    "expectation_consistency": expectation_consistency,
                }
            )

        return pd.DataFrame(features)

    def compute_analyst_features(
        self,
        trade_date: str,
        lookback_days: int = 120,
    ) -> pd.DataFrame:
        """兼容旧接口：分析师预期因子

        旧接口使用天数回溯，这里按近似月份换算。
        """
        lookback_months = max(1, int(round(lookback_days / 30)))
        return self.compute_analyst_expectation_features(trade_date=trade_date, lookback_months=lookback_months)

    # ==================== 统一入口 ====================

    def compute_all_features(
        self,
        trade_date: str,
        moneyflow_lookback: int = 20,
        northbound_lookback: int = 60,
        margin_lookback: int = 20,
        analyst_lookback_months: int = 3,
    ) -> pd.DataFrame:
        """计算所有高 Alpha 因子

        Args:
            trade_date: 交易日期（YYYYMMDD）
            moneyflow_lookback: 资金流回溯天数
            northbound_lookback: 北向资金回溯天数
            margin_lookback: 融资融券回溯天数
            analyst_lookback_months: 分析师预期回溯月数

        Returns:
            DataFrame，包含所有高 Alpha 因子
        """
        # 1. 资金流因子
        moneyflow_features = self.compute_moneyflow_features(trade_date, moneyflow_lookback)

        # 2. 北向资金因子
        northbound_features = self.compute_northbound_features(trade_date, northbound_lookback)

        # 3. 融资融券因子（市场级别，需要广播到个股）
        margin_features = self.compute_margin_features(trade_date, margin_lookback)

        # 4. 分析师预期因子
        analyst_features = self.compute_analyst_expectation_features(trade_date, analyst_lookback_months)

        # 合并所有因子
        if moneyflow_features.empty and northbound_features.empty and analyst_features.empty:
            return pd.DataFrame()

        # 以资金流因子为基础（覆盖面最广）
        if not moneyflow_features.empty:
            result = moneyflow_features
        elif not northbound_features.empty:
            result = northbound_features
        else:
            result = analyst_features

        # 依次合并其他因子
        if not northbound_features.empty and "ts_code" in result.columns:
            result = result.merge(northbound_features, on="ts_code", how="left")

        if not analyst_features.empty and "ts_code" in result.columns:
            result = result.merge(analyst_features, on="ts_code", how="left")

        # 融资融券因子是市场级别，广播到所有股票
        if not margin_features.empty and "ts_code" in result.columns:
            for col in margin_features.columns:
                if col != "trade_date":
                    result[col] = margin_features[col].iloc[0]

        return result
