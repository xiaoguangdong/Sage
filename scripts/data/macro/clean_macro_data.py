#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
宏观数据清洗和特征工程

功能：
1. 加载Tushare宏观数据
2. 加载NBS行业数据
3. 数据对齐和清洗
4. 计算衍生特征
5. 输出整合后的特征数据
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.macro.paths import MACRO_DIR
from scripts.data._shared.runtime import get_data_path


# 修复pandas版本兼容性
try:
    # 新版本pandas (>=2.2)
    pd.date_range.__module__
except:
    pass


class MacroDataProcessor:
    """
    宏观数据处理器
    
    功能：
    1. 加载各类数据源
    2. 数据清洗和标准化
    3. 特征工程
    4. 数据对齐
    """
    
    def __init__(self, data_dir: str = None):
        """
        初始化处理器
        
        Args:
            data_dir: 数据目录
        """
        if data_dir:
            self.data_dir = data_dir
        else:
            macro_dir = Path(MACRO_DIR)
            legacy_dir = PROJECT_ROOT / "data" / "tushare" / "macro"
            self.data_dir = str(macro_dir if macro_dir.exists() else legacy_dir)
        self.macro_data = None
        self.industry_data = None
        self.northbound_data = None
        self.processed_dir = str(get_data_path("processed"))
    
    def load_tushare_macro(self) -> pd.DataFrame:
        """
        加载Tushare宏观数据
        
        Returns:
            DataFrame: 宏观数据
        """
        # 加载CPI
        cpi_path = os.path.join(self.data_dir, 'tushare_cpi.parquet')
        if os.path.exists(cpi_path):
            cpi = pd.read_parquet(cpi_path)
            # 处理月份格式
            cpi['date'] = pd.to_datetime(cpi['month'].astype(str), format='%Y%m')
            cpi = cpi[['date', 'nt_yoy']].rename(columns={'nt_yoy': 'cpi_yoy'})
        else:
            cpi = pd.DataFrame(columns=['date', 'cpi_yoy'])
        
        # 加载PPI
        ppi_path = os.path.join(self.data_dir, 'tushare_ppi.parquet')
        if os.path.exists(ppi_path):
            ppi = pd.read_parquet(ppi_path)
            ppi['date'] = pd.to_datetime(ppi['month'].astype(str), format='%Y%m')
            ppi = ppi[['date', 'ppi_yoy']]
        else:
            ppi = pd.DataFrame(columns=['date', 'ppi_yoy'])
        
        # 加载PMI
        pmi_path = os.path.join(self.data_dir, 'tushare_pmi.parquet')
        if os.path.exists(pmi_path):
            pmi = pd.read_parquet(pmi_path)
            # 检查列名
            if 'MONTH' in pmi.columns:
                pmi['date'] = pd.to_datetime(pmi['MONTH'].astype(str), format='%Y%m')
            elif 'month' in pmi.columns:
                pmi['date'] = pd.to_datetime(pmi['month'].astype(str), format='%Y%m')
            else:
                # 尝试从行索引或其他列获取
                pmi['date'] = pd.to_datetime(pmi.index, format='%Y%m')
            
            # 使用PMI010000（制造业PMI）作为主要指标
            if 'PMI010000' in pmi.columns:
                pmi = pmi[['date', 'PMI010000']].rename(columns={'PMI010000': 'pmi'})
            elif 'markit_pmi' in pmi.columns:
                pmi = pmi[['date', 'markit_pmi']].rename(columns={'markit_pmi': 'pmi'})
            else:
                # 取第一个PMI列
                pmi_cols = [col for col in pmi.columns if col.startswith('PMI')]
                if pmi_cols:
                    pmi = pmi[['date', pmi_cols[0]]].rename(columns={pmi_cols[0]: 'pmi'})
                else:
                    pmi = pd.DataFrame(columns=['date', 'pmi'])
        else:
            pmi = pd.DataFrame(columns=['date', 'pmi'])
        
        # 加载10年期国债收益率
        yield_df = pd.DataFrame(columns=['date', 'yield_10y'])
        yield_path = os.path.join(self.data_dir, 'tushare_yield_10y.parquet')
        if os.path.exists(yield_path):
            temp = pd.read_parquet(yield_path)
            if 'date' in temp.columns and 'yield_10y' in temp.columns:
                temp['date'] = pd.to_datetime(temp['date'])
                yield_df = temp[['date', 'yield_10y']]
        else:
            yield_alt = os.path.join(self.data_dir, 'yield_10y.parquet')
            if os.path.exists(yield_alt):
                temp = pd.read_parquet(yield_alt)
                if 'trade_date' in temp.columns and 'yield' in temp.columns:
                    temp['date'] = pd.to_datetime(temp['trade_date'].astype(str))
                    temp = temp.rename(columns={'yield': 'yield_10y'})
                    yield_df = temp[['date', 'yield_10y']]
        
        # 合并数据
        macro = cpi.merge(ppi, on='date', how='outer')
        macro = macro.merge(pmi, on='date', how='outer')
        macro = macro.merge(yield_df, on='date', how='outer')
        
        # 按日期排序
        macro = macro.sort_values('date').reset_index(drop=True)
        
        # 加载社融和M2数据
        credit_path = os.path.join(self.data_dir, 'credit_data.parquet')
        if os.path.exists(credit_path):
            credit = pd.read_parquet(credit_path)
            if 'date' in credit.columns:
                credit['date'] = pd.to_datetime(credit['date'])
            if 'credit_growth' in credit.columns:
                macro = macro.merge(credit[['date', 'credit_growth']], on='date', how='left')
        
        money_path = os.path.join(self.data_dir, 'money_supply.parquet')
        if os.path.exists(money_path):
            money = pd.read_parquet(money_path)
            if 'date' in money.columns:
                money['date'] = pd.to_datetime(money['date'])
            if 'm1_yoy' in money.columns and 'm2_yoy' in money.columns:
                money['m1_m2_spread'] = money['m1_yoy'] - money['m2_yoy']
                macro = macro.merge(money[['date', 'm1_yoy', 'm2_yoy', 'm1_m2_spread']], on='date', how='left')

        # 确保关键列存在
        for col in ['credit_growth', 'm1_yoy', 'm2_yoy', 'm1_m2_spread']:
            if col not in macro.columns:
                macro[col] = np.nan
        
        self.macro_data = macro
        print(f"加载Tushare宏观数据: {len(macro)}条记录")
        
        return macro
    
    def load_nbs_industry(self) -> pd.DataFrame:
        """
        加载NBS行业数据
        
        Returns:
            DataFrame: 行业数据
        """
        industry_list = []
        
        # 加载分行业PPI数据
        ppi_files = [
            'nbs_ppi_industry_2020.csv',
            'nbs_ppi_industry_202512.csv'
        ]
        
        for ppi_file in ppi_files:
            ppi_path = os.path.join(self.data_dir, ppi_file)
            if os.path.exists(ppi_path):
                ppi_df = pd.read_csv(ppi_path)
                if 'date' in ppi_df.columns and 'industry' in ppi_df.columns:
                    industry_list.append(ppi_df)
        
        # 加载分行业FAI数据
        fai_files = [
            'nbs_fai_industry_2020.csv',
            'nbs_fai_industry_202512.csv'
        ]
        
        for fai_file in fai_files:
            fai_path = os.path.join(self.data_dir, fai_file)
            if os.path.exists(fai_path):
                fai_df = pd.read_csv(fai_path)
                if 'date' in fai_df.columns and 'industry' in fai_df.columns:
                    industry_list.append(fai_df)
        
        # 加载分行业Output数据
        output_files = [
            'nbs_output_2020.csv',
            'nbs_output_202512.csv'
        ]
        
        for output_file in output_files:
            output_path = os.path.join(self.data_dir, output_file)
            if os.path.exists(output_path):
                output_df = pd.read_csv(output_path)
                if 'date' in output_df.columns and 'industry' in output_df.columns:
                    industry_list.append(output_df)
        
        # 合并所有行业数据
        if industry_list:
            industry_df = pd.concat(industry_list, ignore_index=True)
            # 统一日期格式
            industry_df['date'] = pd.to_datetime(industry_df['date'])
            industry_df = industry_df.sort_values(['industry', 'date']).reset_index(drop=True)
        else:
            industry_df = pd.DataFrame()
        
        self.industry_data = industry_df
        print(f"加载NBS行业数据: {len(industry_df)}条记录")
        
        return industry_df

    def load_industry_features_sw_l1(self) -> pd.DataFrame:
        """
        加载申万一级行业聚合特征（优先使用已生成文件）
        """
        processed_path = os.path.join(self.processed_dir, 'industry_features_sw_l1.parquet')
        if os.path.exists(processed_path):
            df = pd.read_parquet(processed_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df

        # fallback: 合并ppi/fai/output
        ppi_path = os.path.join(self.data_dir, 'sw_l1_ppi_yoy_202512.csv')
        fai_path = os.path.join(self.processed_dir, 'fai_sw_industry.parquet')
        output_path = os.path.join(self.processed_dir, 'nbs_industrial_aligned.parquet')

        dfs: List[pd.DataFrame] = []
        if os.path.exists(ppi_path):
            ppi_df = pd.read_csv(ppi_path)
            ppi_df['date'] = pd.to_datetime(ppi_df['date'])
            if 'sw_ppi_yoy' in ppi_df.columns:
                ppi_df = ppi_df.rename(columns={'sw_ppi_yoy': 'ppi_yoy'})
            dfs.append(ppi_df[['sw_industry', 'date', 'ppi_yoy']])

        if os.path.exists(fai_path):
            fai_df = pd.read_parquet(fai_path)
            fai_df['date'] = pd.to_datetime(fai_df['date'])
            dfs.append(fai_df[['sw_industry', 'date', 'fai_yoy', 'fai_mom']])

        if os.path.exists(output_path):
            out_df = pd.read_parquet(output_path)
            out_df['date'] = pd.to_datetime(out_df['date'])
            keep_cols = ['sw_industry', 'date']
            for col in ['output_yoy', 'output_mom', 'output_value', 'fai_mom', 'fai_value']:
                if col in out_df.columns:
                    keep_cols.append(col)
            out_df = out_df[keep_cols]
            dfs.append(out_df)

        if not dfs:
            return pd.DataFrame()

        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on=['sw_industry', 'date'], how='outer')
        return merged.sort_values(['sw_industry', 'date']).reset_index(drop=True)

    def load_sw_industry(self) -> pd.DataFrame:
        """
        加载申万行业数据（已映射）
        
        Returns:
            DataFrame: 申万行业数据
        """
        # 加载L1行业PPI数据
        l1_ppi_path = os.path.join(self.data_dir, 'sw_l1_ppi_yoy_202512.csv')
        if os.path.exists(l1_ppi_path):
            sw_l1 = pd.read_csv(l1_ppi_path)
            sw_l1['date'] = pd.to_datetime(sw_l1['date'])
            sw_l1 = sw_l1.rename(columns={'sw_industry': 'industry'})
        else:
            sw_l1 = pd.DataFrame()
        
        # 加载L2行业PPI数据
        l2_ppi_path = os.path.join(self.data_dir, 'sw_l2_ppi_yoy_202512.csv')
        if os.path.exists(l2_ppi_path):
            sw_l2 = pd.read_csv(l2_ppi_path)
            sw_l2['date'] = pd.to_datetime(sw_l2['date'])
            sw_l2 = sw_l2.rename(columns={'sw_industry': 'industry'})
        else:
            sw_l2 = pd.DataFrame()
        
        # 合并L1和L2数据
        if len(sw_l1) > 0 and len(sw_l2) > 0:
            sw_industry = pd.concat([sw_l1, sw_l2], ignore_index=True)
        elif len(sw_l1) > 0:
            sw_industry = sw_l1
        elif len(sw_l2) > 0:
            sw_industry = sw_l2
        else:
            sw_industry = pd.DataFrame()
        
        if len(sw_industry) > 0:
            sw_industry = sw_industry.sort_values(['industry', 'date']).reset_index(drop=True)
        
        print(f"加载申万行业数据: {len(sw_industry)}条记录")
        
        return sw_industry

    def _get_tushare_root(self) -> Path:
        return Path(self.data_dir).parent

    def load_sw_turnover_rate(self, sw_list: List[str]) -> pd.DataFrame:
        """
        计算申万行业月度换手率（基于sw_daily_all）
        """
        path = self._get_tushare_root() / "sectors" / "sw_daily_all.parquet"
        if not path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(path)
        if df.empty:
            return df

        df = df[df["name"].isin(sw_list)].copy()
        if df.empty:
            return pd.DataFrame()

        df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str))
        df = df.sort_values(["name", "trade_date"])
        df["turnover_rate"] = np.nan
        if "amount" in df.columns and "float_mv" in df.columns:
            df["turnover_rate"] = pd.to_numeric(df["amount"], errors="coerce") / pd.to_numeric(df["float_mv"], errors="coerce").replace(0, np.nan)

        df["date"] = df["trade_date"].dt.to_period("M").dt.to_timestamp()
        df_monthly = df.groupby(["name", "date"]).tail(1).copy()
        df_monthly = df_monthly.rename(columns={"name": "sw_industry"})
        return df_monthly[["sw_industry", "date", "turnover_rate"]]

    def load_sw_stock_industry_map(self) -> pd.DataFrame:
        """
        构建股票 -> 申万一级行业映射（基于指数成分 + L2/L1分类）
        """
        sectors_dir = self._get_tushare_root() / "sectors"
        members_path = sectors_dir / "all_index_members.csv"
        l2_path = sectors_dir / "SW2021_L2_classify.csv"
        l1_path = sectors_dir / "SW2021_L1_classify.csv"

        if not (members_path.exists() and l2_path.exists() and l1_path.exists()):
            return pd.DataFrame()

        members = pd.read_csv(members_path)
        if "industry_name" in members.columns:
            members = members.rename(columns={"industry_name": "industry_name_l2"})
        l2 = pd.read_csv(l2_path)
        l1 = pd.read_csv(l1_path)

        members["in_date"] = pd.to_datetime(members["in_date"], errors="coerce")
        l2_map = l2[["index_code", "parent_code"]]
        l1_map = l1[["industry_code", "industry_name"]].rename(columns={"industry_code": "parent_code"})

        merged = members.merge(l2_map, on="index_code", how="left")
        merged = merged.merge(l1_map, on="parent_code", how="left")

        merged = merged.sort_values("in_date").dropna(subset=["industry_name"])
        latest = merged.groupby("con_code").tail(1)
        return latest.rename(columns={"con_code": "ts_code", "industry_name": "sw_industry"})[["ts_code", "sw_industry"]]

    def load_industry_rev_yoy(self, stock_map: pd.DataFrame) -> pd.DataFrame:
        """
        汇总行业营收同比（rev_yoy）
        """
        if stock_map is None or stock_map.empty:
            return pd.DataFrame()

        fundamental_dir = self._get_tushare_root() / "fundamental"
        if not fundamental_dir.exists():
            return pd.DataFrame()

        frames = []
        for path in sorted(fundamental_dir.glob("fina_indicator_*.parquet")):
            df = pd.read_parquet(path, columns=["ts_code", "end_date", "tr_yoy"])
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        data = pd.concat(frames, ignore_index=True)
        data["end_date"] = pd.to_datetime(data["end_date"].astype(str), errors="coerce")
        data = data.merge(stock_map, on="ts_code", how="inner")
        data = data.dropna(subset=["sw_industry", "end_date"])
        if data.empty:
            return pd.DataFrame()

        data["date"] = data["end_date"].dt.to_period("M").dt.to_timestamp()
        grouped = data.groupby(["sw_industry", "date"])["tr_yoy"].mean().reset_index()
        grouped = grouped.rename(columns={"tr_yoy": "rev_yoy"})
        return grouped

    def load_sw_valuation_features(self, sw_list: List[str]) -> pd.DataFrame:
        """
        计算行业估值与RPS特征（按月聚合）
        """
        valuation_path = os.path.join(self.data_dir, 'sw_valuation.parquet')
        if not os.path.exists(valuation_path):
            return pd.DataFrame()

        df = pd.read_parquet(valuation_path)
        if df.empty or 'name' not in df.columns:
            return pd.DataFrame()

        df = df[df['name'].isin(sw_list)].copy()
        if df.empty:
            return pd.DataFrame()

        df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
        df = df.sort_values(['name', 'trade_date'])
        df = df.rename(columns={'name': 'sw_industry'})

        if 'close' in df.columns:
            df['ret_120'] = df.groupby('sw_industry')['close'].pct_change(120)
            df['rps_120'] = df.groupby('trade_date')['ret_120'].rank(pct=True) * 100
        else:
            df['rps_120'] = np.nan

        df['date'] = df['trade_date'].dt.to_period('M').dt.to_timestamp()
        df_monthly = df.groupby(['sw_industry', 'date']).tail(1).copy()

        def _expanding_percentile(series: pd.Series) -> pd.Series:
            values: List[float] = []
            result: List[float] = []
            for val in series:
                if pd.isna(val):
                    result.append(np.nan)
                    continue
                values.append(val)
                valid = [v for v in values if pd.notna(v)]
                rank = pd.Series(valid).rank(pct=True).iloc[-1] * 100
                result.append(rank)
            return pd.Series(result, index=series.index)

        if 'pb' in df_monthly.columns:
            df_monthly['pb_percentile'] = df_monthly.groupby('sw_industry')['pb'].transform(_expanding_percentile)
        else:
            df_monthly['pb_percentile'] = np.nan

        if 'pe' in df_monthly.columns:
            df_monthly['pe_percentile'] = df_monthly.groupby('sw_industry')['pe'].transform(_expanding_percentile)
        else:
            df_monthly['pe_percentile'] = np.nan

        return df_monthly[['sw_industry', 'date', 'pb_percentile', 'pe_percentile', 'rps_120']]
    
    def load_northbound(self) -> pd.DataFrame:
        """
        加载北向资金数据
        
        Returns:
            DataFrame: 北向资金数据
        """
        northbound_dir = os.path.join(self.data_dir, '..', 'northbound')
        northbound_files = []
        
        if os.path.exists(northbound_dir):
            for file in os.listdir(northbound_dir):
                if file.endswith('.parquet'):
                    northbound_files.append(os.path.join(northbound_dir, file))
        
        if northbound_files:
            northbound_dfs = []
            for file in northbound_files:
                df = pd.read_parquet(file)
                if 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    northbound_dfs.append(df)
            
            if northbound_dfs:
                northbound = pd.concat(northbound_dfs, ignore_index=True)
                northbound = northbound.sort_values('trade_date').reset_index(drop=True)
            else:
                northbound = pd.DataFrame()
        else:
            northbound = pd.DataFrame()
        
        self.northbound_data = northbound
        print(f"加载北向资金数据: {len(northbound)}条记录")
        
        return northbound

    def load_northbound_features(self) -> pd.DataFrame:
        """
        生成北向行业配置比例信号（Bollinger）
        """
        industry_path = os.path.join(self.data_dir, '..', 'northbound', 'industry_northbound_flow.parquet')
        if not os.path.exists(industry_path):
            legacy_path = PROJECT_ROOT / "data" / "tushare" / "northbound" / "industry_northbound_flow.parquet"
            if legacy_path.exists():
                industry_path = str(legacy_path)
            else:
                return pd.DataFrame()

        df = pd.read_parquet(industry_path)
        if df.empty:
            return df

        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values(['industry_code', 'trade_date'])

        if 'ratio' in df.columns and df['ratio'].notna().any():
            df['industry_ratio'] = pd.to_numeric(df['ratio'], errors='coerce')
        else:
            df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
            total_vol = df.groupby('trade_date')['vol'].transform('sum')
            df['industry_ratio'] = df['vol'] / total_vol.replace(0, np.nan)

        df['ratio_ma'] = df.groupby('industry_code')['industry_ratio'].transform(
            lambda s: s.rolling(window=20).mean()
        )
        df['ratio_std'] = df.groupby('industry_code')['industry_ratio'].transform(
            lambda s: s.rolling(window=20).std()
        )
        df['upper_band'] = df['ratio_ma'] + 2.0 * df['ratio_std']
        df['lower_band'] = df['ratio_ma'] - 2.0 * df['ratio_std']

        df['northbound_signal'] = 0
        df.loc[df['industry_ratio'] > df['upper_band'], 'northbound_signal'] = 1
        df.loc[df['industry_ratio'] < df['lower_band'], 'northbound_signal'] = -1

        df['sw_industry'] = df['industry_name']
        return df[['sw_industry', 'industry_name', 'trade_date', 'industry_ratio', 'northbound_signal']]
    
    def clean_macro_data(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗宏观数据
        
        Args:
            macro_df: 宏观数据
        
        Returns:
            DataFrame: 清洗后的数据
        """
        df = macro_df.copy()
        
        # 1. 前向填充缺失值
        df = df.ffill()
        
        # 2. 计算变化率
        for col in ['cpi_yoy', 'ppi_yoy', 'pmi', 'yield_10y']:
            if col in df.columns:
                df[f'{col}_delta'] = df[col].diff()
        
        # 3. 计算移动平均
        for col in ['cpi_yoy', 'ppi_yoy', 'pmi']:
            if col in df.columns:
                df[f'{col}_ma3'] = df[col].rolling(3, min_periods=1).mean()
                df[f'{col}_ma6'] = df[col].rolling(6, min_periods=1).mean()
        
        # 4. 计算相对位置
        for col in ['cpi_yoy', 'ppi_yoy', 'yield_10y']:
            if col in df.columns:
                rolling_min = df[col].rolling(12, min_periods=1).min()
                rolling_max = df[col].rolling(12, min_periods=1).max()
                df[f'{col}_percentile'] = (df[col] - rolling_min) / (rolling_max - rolling_min + 1e-8)
        
        return df
    
    def clean_industry_data(self, industry_df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗行业数据
        
        Args:
            industry_df: 行业数据
        
        Returns:
            DataFrame: 清洗后的数据
        """
        df = industry_df.copy()
        
        # 1. 前向填充缺失值
        df = df.ffill()
        
        group_col = 'industry' if 'industry' in df.columns else 'sw_industry'

        # 2. 计算变化率
        numeric_cols = ['fai_yoy', 'output_yoy', 'sw_ppi_yoy']
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_delta'] = df.groupby(group_col)[col].diff()
        
        # 3. 计算移动平均
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_ma3'] = df.groupby(group_col)[col].transform(
                    lambda x: x.rolling(3, min_periods=1).mean()
                )
        
        return df
    
    def add_market_data(self, industry_df: pd.DataFrame, sw_list: List[str]) -> pd.DataFrame:
        """
        添加估值与RPS特征（无则留空）
        """
        if industry_df is None or industry_df.empty:
            return industry_df

        valuation_df = self.load_sw_valuation_features(sw_list)
        turnover_df = self.load_sw_turnover_rate(sw_list)
        stock_map = self.load_sw_stock_industry_map()
        rev_yoy_df = self.load_industry_rev_yoy(stock_map)

        if valuation_df.empty:
            valuation_df = pd.DataFrame()

        result = industry_df.copy()
        if not valuation_df.empty:
            result = result.merge(
                valuation_df,
                on=['sw_industry', 'date'],
                how='left'
            )
        if not turnover_df.empty:
            result = result.merge(
                turnover_df,
                on=['sw_industry', 'date'],
                how='left'
            )
        if not rev_yoy_df.empty:
            result = result.merge(
                rev_yoy_df,
                on=['sw_industry', 'date'],
                how='left'
            )
        return result
    
    def process_all(self) -> Dict[str, pd.DataFrame]:
        """
        处理所有数据
        
        Returns:
            Dict: 包含所有处理后的数据
        """
        print("=" * 80)
        print("开始处理宏观数据")
        print("=" * 80)
        
        # 1. 加载数据
        macro = self.load_tushare_macro()
        sw_industry = self.load_industry_features_sw_l1()
        northbound = self.load_northbound_features()
        
        # 2. 清洗数据
        macro_cleaned = self.clean_macro_data(macro)
        industry_cleaned = self.clean_industry_data(sw_industry)

        # 3. 添加估值/RPS
        try:
            import yaml  # type: ignore
            with open('config/sw_nbs_mapping.yaml', 'r', encoding='utf-8') as f:
                sw_list = list((yaml.safe_load(f) or {}).get('sw_to_nbs', {}).keys())
        except Exception:
            sw_list = []

        industry_with_market = self.add_market_data(industry_cleaned, sw_list)
        if not industry_with_market.empty:
            industry_with_market = industry_with_market.sort_values(["sw_industry", "date"])
            industry_with_market["rev_yoy"] = industry_with_market.groupby("sw_industry")["rev_yoy"].ffill()
            industry_with_market["turnover_rate"] = industry_with_market.groupby("sw_industry")["turnover_rate"].ffill()

        # 4. 补齐字段
        industry_final = industry_with_market.copy()
        if 'sw_ppi_yoy' not in industry_final.columns and 'ppi_yoy' in industry_final.columns:
            industry_final['sw_ppi_yoy'] = industry_final['ppi_yoy']

        if 'inventory_yoy' not in industry_final.columns:
            industry_final['inventory_yoy'] = np.nan
        if 'output_yoy' in industry_final.columns and 'rev_yoy' in industry_final.columns:
            mask = industry_final['inventory_yoy'].isna() & industry_final['output_yoy'].notna() & industry_final['rev_yoy'].notna()
            industry_final.loc[mask, 'inventory_yoy'] = industry_final.loc[mask, 'output_yoy'] - industry_final.loc[mask, 'rev_yoy']

        required_cols = [
            'sw_industry', 'date', 'sw_ppi_yoy', 'fai_yoy',
            'pb_percentile', 'pe_percentile', 'turnover_rate', 'rps_120',
            'inventory_yoy', 'rev_yoy'
        ]
        for col in required_cols:
            if col not in industry_final.columns:
                industry_final[col] = np.nan
        
        print("\n" + "=" * 80)
        print("数据处理完成")
        print("=" * 80)
        print(f"宏观数据: {len(macro_cleaned)}条")
        print(f"行业数据: {len(industry_final)}条")
        print(f"北向资金: {len(northbound)}条")
        
        return {
            'macro': macro_cleaned,
            'industry': industry_final,
            'northbound': northbound
        }


def main():
    """测试数据处理器"""
    processor = MacroDataProcessor()
    
    # 处理数据
    data = processor.process_all()
    
    # 输出样例数据
    print("\n宏观数据样例:")
    print(data['macro'].head())
    
    print("\n行业数据样例:")
    print(data['industry'].head())
    
    # 保存数据
    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)
    
    data['macro'].to_parquet(f'{output_dir}/macro_features.parquet', index=False)
    data['industry'].to_parquet(f'{output_dir}/industry_features.parquet', index=False)
    
    if len(data['northbound']) > 0:
        data['northbound'].to_parquet(f'{output_dir}/northbound_features.parquet', index=False)

    # 对齐审计报告
    def _coverage(df: pd.DataFrame, required: List[str]) -> Dict:
        present = [c for c in required if c in df.columns]
        missing = [c for c in required if c not in df.columns]
        null_ratio = {}
        for col in present:
            if len(df) == 0:
                null_ratio[col] = 1.0
            else:
                null_ratio[col] = float(df[col].isna().mean())
        return {"missing": missing, "null_ratio": null_ratio}

    report = {
        "macro": _coverage(data['macro'], ['date', 'credit_growth', 'pmi', 'cpi_yoy', 'yield_10y']),
        "industry": _coverage(data['industry'], ['sw_industry', 'date', 'sw_ppi_yoy', 'fai_yoy',
                                                'pb_percentile', 'pe_percentile', 'turnover_rate', 'rps_120',
                                                'inventory_yoy', 'rev_yoy']),
        "northbound": _coverage(data['northbound'], ['sw_industry', 'trade_date', 'northbound_signal', 'industry_ratio']),
    }
    report_path = os.path.join(output_dir, 'macro_alignment_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n对齐审计报告已保存: {report_path}")
    
    print(f"\n数据已保存到 {output_dir}")


if __name__ == '__main__':
    main()
