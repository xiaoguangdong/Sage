#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
长周期基本面特征计算模块

用于波段选股策略的价值股/成长股特征
基于Tushare数据，计算TTM口径的财务指标

特征列表：
1. 研发费用率（TTM）
2. ROE 5年均值
3. 营收/利润3年CAGR
4. 连续分红年数
5. 利息保障倍数
6. 机构持仓变化
7. 市占率（预留接口，需外部数据补充）
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class LongTermFundamentalFeatures:
    """长周期基本面特征计算器

    设计原则：
    - 所有指标使用TTM（滚动12个月）口径，避免季节性波动
    - 严格遵守T+2延迟，避免未来函数
    - 支持行业内标准化（z-score）
    """

    def __init__(self, data_root: Path):
        """初始化

        Args:
            data_root: Tushare数据根目录
        """
        self.data_root = Path(data_root)
        self.income_dir = self.data_root / "fundamental" / "income"
        self.balance_dir = self.data_root / "fundamental" / "balancesheet"
        self.cashflow_dir = self.data_root / "fundamental" / "cashflow"
        self.fina_indicator_dir = self.data_root / "fundamental"
        self.daily_basic_dir = self.data_root / "daily_basic"
        self.dividend_dir = self.data_root / "fundamental" / "dividend"

    def calculate_rd_expense_ratio_ttm(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """计算研发费用率（TTM口径）

        公式：研发费用率 = 最近4个季度累计研发费用 / 最近4个季度累计营收 × 100%

        Args:
            start_date: 起始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            DataFrame with columns: [ts_code, end_date, rd_expense_ratio_ttm]
        """
        # 加载利润表数据
        income_files = sorted(self.income_dir.glob("income_*.parquet"))
        if not income_files:
            raise FileNotFoundError(f"未找到利润表数据: {self.income_dir}")

        df_list = []
        for file in income_files:
            df = pd.read_parquet(file)
            df_list.append(df)

        df_income = pd.concat(df_list, ignore_index=True)

        # 筛选需要的字段和时间范围
        df_income = df_income[["ts_code", "end_date", "rd_exp", "revenue"]].copy()
        df_income["end_date"] = pd.to_datetime(df_income["end_date"], format="%Y%m%d")
        df_income = df_income[
            (df_income["end_date"] >= pd.to_datetime(start_date, format="%Y%m%d"))
            & (df_income["end_date"] <= pd.to_datetime(end_date, format="%Y%m%d"))
        ]

        # 按股票代码和日期排序
        df_income = df_income.sort_values(["ts_code", "end_date"])

        # 计算TTM（滚动4个季度）
        df_income["rd_exp_ttm"] = (
            df_income.groupby("ts_code")["rd_exp"]
            .rolling(window=4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )

        df_income["revenue_ttm"] = (
            df_income.groupby("ts_code")["revenue"]
            .rolling(window=4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )

        # 计算研发费用率
        df_income["rd_expense_ratio_ttm"] = (
            df_income["rd_exp_ttm"] / df_income["revenue_ttm"] * 100
        )

        # 处理异常值
        df_income["rd_expense_ratio_ttm"] = df_income["rd_expense_ratio_ttm"].clip(
            lower=0, upper=100
        )

        # 返回结果
        result = df_income[["ts_code", "end_date", "rd_expense_ratio_ttm"]].copy()
        result["end_date"] = result["end_date"].dt.strftime("%Y%m%d")

        return result.dropna()

    def calculate_roe_5y_avg(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """计算ROE 5年滚动均值

        公式：ROE_5Y_AVG = mean(ROE_TTM[-20个季度])

        Args:
            start_date: 起始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            DataFrame with columns: [ts_code, end_date, roe_5y_avg]
        """
        # 加载财务指标数据
        fina_files = sorted(self.fina_indicator_dir.glob("fina_indicator_*.parquet"))
        if not fina_files:
            raise FileNotFoundError(f"未找到财务指标数据: {self.fina_indicator_dir}")

        df_list = []
        for file in fina_files:
            df = pd.read_parquet(file)
            df_list.append(df)

        df_fina = pd.concat(df_list, ignore_index=True)

        # 筛选需要的字段
        df_fina = df_fina[["ts_code", "end_date", "roe"]].copy()
        df_fina["end_date"] = pd.to_datetime(df_fina["end_date"], format="%Y%m%d")

        # 按股票代码和日期排序（使用全部历史数据）
        df_fina = df_fina.sort_values(["ts_code", "end_date"])

        # 计算5年滚动均值（20个季度）
        df_fina["roe_5y_avg"] = (
            df_fina.groupby("ts_code")["roe"]
            .rolling(window=20, min_periods=12)  # 至少3年数据
            .mean()
            .reset_index(level=0, drop=True)
        )

        # 最后筛选目标日期范围
        df_fina = df_fina[
            (df_fina["end_date"] >= pd.to_datetime(start_date, format="%Y%m%d"))
            & (df_fina["end_date"] <= pd.to_datetime(end_date, format="%Y%m%d"))
        ]

        # 返回结果
        result = df_fina[["ts_code", "end_date", "roe_5y_avg"]].copy()
        result["end_date"] = result["end_date"].dt.strftime("%Y%m%d")

        return result.dropna()

    def calculate_revenue_cagr_3y(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """计算营收3年CAGR

        公式：CAGR = (最新年度营收 / 3年前年度营收)^(1/3) - 1

        Args:
            start_date: 起始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            DataFrame with columns: [ts_code, end_date, revenue_cagr_3y]
        """
        # 加载利润表数据
        income_files = sorted(self.income_dir.glob("income_*.parquet"))
        if not income_files:
            raise FileNotFoundError(f"未找到利润表数据: {self.income_dir}")

        df_list = []
        for file in income_files:
            df = pd.read_parquet(file)
            df_list.append(df)

        df_income = pd.concat(df_list, ignore_index=True)

        # 筛选年报数据（1231结尾）
        df_income = df_income[["ts_code", "end_date", "revenue"]].copy()
        df_income = df_income[df_income["end_date"].astype(str).str.endswith("1231")]
        df_income["end_date"] = pd.to_datetime(df_income["end_date"], format="%Y%m%d")

        # 按股票代码和日期排序（使用全部历史数据）
        df_income = df_income.sort_values(["ts_code", "end_date"])

        # 计算3年前的营收
        df_income["revenue_3y_ago"] = df_income.groupby("ts_code")["revenue"].shift(3)

        # 计算CAGR
        df_income["revenue_cagr_3y"] = (
            (df_income["revenue"] / df_income["revenue_3y_ago"]) ** (1 / 3) - 1
        ) * 100

        # 最后筛选目标日期范围
        df_income = df_income[
            (df_income["end_date"] >= pd.to_datetime(start_date, format="%Y%m%d"))
            & (df_income["end_date"] <= pd.to_datetime(end_date, format="%Y%m%d"))
        ]

        # 返回结果
        result = df_income[["ts_code", "end_date", "revenue_cagr_3y"]].copy()
        result["end_date"] = result["end_date"].dt.strftime("%Y%m%d")

        return result.dropna()

    def calculate_profit_cagr_3y(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """计算净利润3年CAGR

        公式：CAGR = (最新年度净利润 / 3年前年度净利润)^(1/3) - 1

        Args:
            start_date: 起始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            DataFrame with columns: [ts_code, end_date, profit_cagr_3y]
        """
        # 加载利润表数据
        income_files = sorted(self.income_dir.glob("income_*.parquet"))
        if not income_files:
            raise FileNotFoundError(f"未找到利润表数据: {self.income_dir}")

        df_list = []
        for file in income_files:
            df = pd.read_parquet(file)
            df_list.append(df)

        df_income = pd.concat(df_list, ignore_index=True)

        # 筛选年报数据（1231结尾）
        df_income = df_income[["ts_code", "end_date", "n_income"]].copy()
        df_income = df_income[df_income["end_date"].astype(str).str.endswith("1231")]
        df_income["end_date"] = pd.to_datetime(df_income["end_date"], format="%Y%m%d")

        # 按股票代码和日期排序（使用全部历史数据）
        df_income = df_income.sort_values(["ts_code", "end_date"])

        # 计算3年前的净利润
        df_income["n_income_3y_ago"] = df_income.groupby("ts_code")["n_income"].shift(3)

        # 计算CAGR（处理负值情况）
        mask = (df_income["n_income"] > 0) & (df_income["n_income_3y_ago"] > 0)
        df_income.loc[mask, "profit_cagr_3y"] = (
            (df_income.loc[mask, "n_income"] / df_income.loc[mask, "n_income_3y_ago"])
            ** (1 / 3)
            - 1
        ) * 100

        # 最后筛选目标日期范围
        df_income = df_income[
            (df_income["end_date"] >= pd.to_datetime(start_date, format="%Y%m%d"))
            & (df_income["end_date"] <= pd.to_datetime(end_date, format="%Y%m%d"))
        ]

        # 返回结果
        result = df_income[["ts_code", "end_date", "profit_cagr_3y"]].copy()
        result["end_date"] = result["end_date"].dt.strftime("%Y%m%d")

        return result.dropna()

    def calculate_consecutive_dividend_years(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """计算连续分红年数

        统计最近N年连续分红的年数

        Args:
            start_date: 起始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            DataFrame with columns: [ts_code, end_date, consecutive_dividend_years]
        """
        # 加载分红数据
        dividend_files = sorted(self.dividend_dir.glob("dividend_*.parquet"))
        if not dividend_files:
            raise FileNotFoundError(f"未找到分红数据: {self.dividend_dir}")

        df_list = []
        for file in dividend_files:
            df = pd.read_parquet(file)
            df_list.append(df)

        df_dividend = pd.concat(df_list, ignore_index=True)

        # 筛选有效分红记录（现金分红 > 0）
        df_dividend = df_dividend[
            ["ts_code", "end_date", "div_proc", "cash_div"]
        ].copy()
        df_dividend = df_dividend[df_dividend["cash_div"] > 0]

        # 提取年份
        df_dividend["year"] = df_dividend["end_date"].astype(str).str[:4].astype(int)

        # 按股票和年份去重
        df_dividend = df_dividend.drop_duplicates(subset=["ts_code", "year"])

        # 按股票代码和年份排序
        df_dividend = df_dividend.sort_values(["ts_code", "year"])

        # 计算连续分红年数
        def count_consecutive_years(group):
            years = sorted(group["year"].values)
            if len(years) == 0:
                return 0

            consecutive = 1
            max_consecutive = 1

            for i in range(1, len(years)):
                if years[i] == years[i - 1] + 1:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 1

            return max_consecutive

        result = (
            df_dividend.groupby("ts_code")
            .apply(count_consecutive_years, include_groups=False)
            .reset_index()
        )
        result.columns = ["ts_code", "consecutive_dividend_years"]

        # 添加end_date（使用最新的分红记录日期）
        latest_dates = (
            df_dividend.groupby("ts_code")["end_date"].max().reset_index()
        )
        result = result.merge(latest_dates, on="ts_code", how="left")

        return result[["ts_code", "end_date", "consecutive_dividend_years"]]

    def calculate_interest_coverage_ratio(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """计算利息保障倍数（TTM口径）

        公式：利息保障倍数 = EBIT_TTM / 利息费用_TTM
        EBIT = 营业利润 + 利息费用

        Args:
            start_date: 起始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            DataFrame with columns: [ts_code, end_date, interest_coverage_ratio]
        """
        # 加载利润表数据
        income_files = sorted(self.income_dir.glob("income_*.parquet"))
        if not income_files:
            raise FileNotFoundError(f"未找到利润表数据: {self.income_dir}")

        df_list = []
        for file in income_files:
            df = pd.read_parquet(file)
            df_list.append(df)

        df_income = pd.concat(df_list, ignore_index=True)

        # 筛选需要的字段
        df_income = df_income[
            ["ts_code", "end_date", "operate_profit", "int_exp"]
        ].copy()
        df_income["end_date"] = pd.to_datetime(df_income["end_date"], format="%Y%m%d")
        df_income = df_income[
            (df_income["end_date"] >= pd.to_datetime(start_date, format="%Y%m%d"))
            & (df_income["end_date"] <= pd.to_datetime(end_date, format="%Y%m%d"))
        ]

        # 按股票代码和日期排序
        df_income = df_income.sort_values(["ts_code", "end_date"])

        # 计算EBIT = 营业利润 + 利息费用
        df_income["ebit"] = df_income["operate_profit"] + df_income["int_exp"]

        # 计算TTM（滚动4个季度）
        df_income["ebit_ttm"] = (
            df_income.groupby("ts_code")["ebit"]
            .rolling(window=4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )

        df_income["int_exp_ttm"] = (
            df_income.groupby("ts_code")["int_exp"]
            .rolling(window=4, min_periods=4)
            .sum()
            .reset_index(level=0, drop=True)
        )

        # 计算利息保障倍数
        df_income["interest_coverage_ratio"] = (
            df_income["ebit_ttm"] / df_income["int_exp_ttm"]
        )

        # 处理异常值（利息费用为0或负数的情况）
        df_income.loc[df_income["int_exp_ttm"] <= 0, "interest_coverage_ratio"] = np.nan

        # 返回结果
        result = df_income[["ts_code", "end_date", "interest_coverage_ratio"]].copy()
        result["end_date"] = result["end_date"].dt.strftime("%Y%m%d")

        return result.dropna()

    def calculate_institutional_holding_change(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """计算机构持仓变化（预留接口）

        需要从top10_floatholders数据中提取机构股东，计算持仓比例变化

        Args:
            start_date: 起始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            DataFrame with columns: [ts_code, end_date, institutional_holding_change]
        """
        # TODO: 实现机构持仓变化计算
        # 需要从data/tushare/holders/目录读取数据
        raise NotImplementedError("机构持仓变化计算待实现")

    def calculate_market_share(
        self,
        start_date: str,
        end_date: str,
        external_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """计算市占率（预留接口，需外部数据）

        Args:
            start_date: 起始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            external_data: 外部市占率数据（爬虫获取）
                格式: [ts_code, end_date, market_share]

        Returns:
            DataFrame with columns: [ts_code, end_date, market_share]
        """
        if external_data is not None:
            return external_data

        # TODO: 实现基于行业内营收排名的替代方案
        raise NotImplementedError("市占率计算需要外部数据补充")

    def calculate_all_features(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """计算所有长周期基本面特征

        Args:
            start_date: 起始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）

        Returns:
            DataFrame with all features
        """
        print(f"计算长周期基本面特征: {start_date} ~ {end_date}")

        # 1. 研发费用率
        print("  [1/7] 计算研发费用率...")
        df_rd = self.calculate_rd_expense_ratio_ttm(start_date, end_date)

        # 2. ROE 5年均值
        print("  [2/7] 计算ROE 5年均值...")
        df_roe = self.calculate_roe_5y_avg(start_date, end_date)

        # 3. 营收3年CAGR
        print("  [3/7] 计算营收3年CAGR...")
        df_revenue_cagr = self.calculate_revenue_cagr_3y(start_date, end_date)

        # 4. 利润3年CAGR
        print("  [4/7] 计算利润3年CAGR...")
        df_profit_cagr = self.calculate_profit_cagr_3y(start_date, end_date)

        # 5. 连续分红年数
        print("  [5/7] 计算连续分红年数...")
        df_dividend = self.calculate_consecutive_dividend_years(start_date, end_date)

        # 6. 利息保障倍数
        print("  [6/7] 计算利息保障倍数...")
        df_interest = self.calculate_interest_coverage_ratio(start_date, end_date)

        # 7. 合并所有特征
        print("  [7/7] 合并所有特征...")
        df_all = df_rd
        for df in [df_roe, df_revenue_cagr, df_profit_cagr, df_dividend, df_interest]:
            df_all = df_all.merge(df, on=["ts_code", "end_date"], how="outer")

        print(f"✓ 完成！共 {len(df_all)} 条记录")
        return df_all
