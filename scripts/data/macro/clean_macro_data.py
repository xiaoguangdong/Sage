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
from functools import reduce

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
        self.processed_dir = str(get_data_path("processed", ensure=True))
    
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
        
        # 加载社融和M2数据
        credit_path = os.path.join(self.data_dir, 'credit_data.parquet')
        credit = pd.DataFrame()
        if os.path.exists(credit_path):
            credit = pd.read_parquet(credit_path)
            if 'date' in credit.columns:
                credit['date'] = pd.to_datetime(credit['date'])
            elif 'month' in credit.columns:
                credit['date'] = pd.to_datetime(credit['month'].astype(str), format='%Y%m', errors='coerce')

            if 'credit_growth' not in credit.columns and 'stk_endval' in credit.columns:
                credit = credit.sort_values('date')
                credit['credit_growth'] = pd.to_numeric(credit['stk_endval'], errors='coerce').pct_change(12) * 100

            if 'credit_growth' in credit.columns:
                credit = credit[['date', 'credit_growth']]
            else:
                credit = pd.DataFrame()
        
        money_path = os.path.join(self.data_dir, 'money_supply.parquet')
        money = pd.DataFrame()
        if os.path.exists(money_path):
            money = pd.read_parquet(money_path)
            if 'date' in money.columns:
                money['date'] = pd.to_datetime(money['date'])
            elif 'month' in money.columns:
                money['date'] = pd.to_datetime(money['month'].astype(str), format='%Y%m', errors='coerce')

            if 'm1_yoy' in money.columns and 'm2_yoy' in money.columns:
                money['m1_m2_spread'] = money['m1_yoy'] - money['m2_yoy']
                money = money[['date', 'm1_yoy', 'm2_yoy', 'm1_m2_spread']]
            else:
                money = pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for frame in [cpi, ppi, pmi, yield_df, credit, money]:
            if frame is not None and not frame.empty:
                frames.append(frame)

        if frames:
            macro = reduce(lambda left, right: left.merge(right, on='date', how='outer'), frames)
        else:
            macro = pd.DataFrame(columns=['date'])

        # 按日期排序
        if not macro.empty:
            macro = macro.sort_values('date').reset_index(drop=True)

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
            ppi_ok = True
            if 'ppi_yoy' in df.columns:
                ppi_ok = df['ppi_yoy'].notna().mean() >= 0.2
            if len(df) >= 100 and ppi_ok:
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
        # 若ppi数据过少，尝试用NBS行业PPI补齐
        if not dfs or len(dfs[0]) < 100:
            ppi_from_nbs = self._build_sw_ppi_from_nbs()
            if not ppi_from_nbs.empty:
                dfs = [ppi_from_nbs]

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

    def _build_sw_ppi_from_nbs(self) -> pd.DataFrame:
        """
        基于NBS行业PPI(上月=100)重建申万行业PPI同比
        """
        nbs_path = os.path.join(self.data_dir, 'nbs_ppi_industry_2020.csv')
        if not os.path.exists(nbs_path):
            legacy = PROJECT_ROOT / "data" / "tushare" / "macro" / "nbs_ppi_industry_2020.csv"
            if legacy.exists():
                nbs_path = str(legacy)
            else:
                return pd.DataFrame()

        df = pd.read_csv(nbs_path)
        if df.empty or 'industry' not in df.columns:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df['ppi_mom'] = pd.to_numeric(df.get('ppi_mom'), errors='coerce')

        def _normalize(name: str) -> str:
            if not isinstance(name, str):
                return ''
            value = name.strip()
            value = value.replace("（", "(").replace("）", ")").replace(" ", "")
            value = value.replace("工业生产者出厂价格指数", "")
            value = value.replace("(上月=100)", "")
            value = value.replace("(上年同期=100)", "")
            value = value.replace("指数", "")
            return value

        df['industry_norm'] = df['industry'].apply(_normalize)
        df = df.sort_values(['industry_norm', 'date'])

        # 由环比指数构建链式指数，再计算同比
        df['ppi_index'] = df.groupby('industry_norm')['ppi_mom'].apply(
            lambda s: (s / 100.0).cumprod() * 100.0
        ).reset_index(level=0, drop=True)
        df['ppi_yoy'] = df.groupby('industry_norm')['ppi_index'].pct_change(12) * 100

        # 加载申万-NBS映射
        mapping_path = PROJECT_ROOT / "config" / "sw_nbs_mapping.yaml"
        if not mapping_path.exists():
            return pd.DataFrame()
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(mapping_path.read_text(encoding='utf-8')) or {}
        except Exception:
            return pd.DataFrame()

        sw_to_nbs = cfg.get('sw_to_nbs') or {}
        nbs_groups = df.groupby('industry_norm')

        rows = []
        for sw, items in sw_to_nbs.items():
            if not items:
                continue
            series_list = []
            for item in items:
                nbs_name = _normalize((item or {}).get('nbs_industry') or '')
                weight = float((item or {}).get('weight', 0) or 0)
                if not nbs_name or weight <= 0:
                    continue
                if nbs_name not in nbs_groups.indices:
                    continue
                part = nbs_groups.get_group(nbs_name)[['date', 'ppi_yoy']].copy()
                part['weight'] = weight
                series_list.append(part)

            if not series_list:
                continue
            combined = pd.concat(series_list, ignore_index=True)
            combined = combined.dropna(subset=['ppi_yoy'])
            if combined.empty:
                continue
            grouped = combined.groupby('date').apply(
                lambda g: (g['ppi_yoy'] * g['weight']).sum() / g['weight'].sum()
            ).reset_index(name='ppi_yoy')
            grouped['sw_industry'] = sw
            rows.append(grouped[['sw_industry', 'date', 'ppi_yoy']])

        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True).sort_values(['sw_industry', 'date']).reset_index(drop=True)

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

        if sw_list:
            filtered = df[df["name"].isin(sw_list)].copy()
            if not filtered.empty:
                df = filtered

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
        l2_paths = [
            sectors_dir / "SW2021_L2_classify.csv",
            sectors_dir / "SW_L2_classify.csv",
        ]
        l1_paths = [
            sectors_dir / "SW2021_L1_classify.csv",
            sectors_dir / "SW_L1_classify.csv",
        ]

        if not members_path.exists():
            return pd.DataFrame()

        members = pd.read_csv(members_path)
        if "industry_name" in members.columns:
            members = members.rename(columns={"industry_name": "industry_name_l2"})
        l2_frames = []
        for idx, path in enumerate(l2_paths):
            if path.exists():
                df = pd.read_csv(path)
                df["priority"] = idx
                l2_frames.append(df)
        if not l2_frames:
            return pd.DataFrame()
        l2 = pd.concat(l2_frames, ignore_index=True)
        l2 = l2.sort_values("priority").drop_duplicates(subset=["index_code"], keep="first")

        l1_frames = []
        for idx, path in enumerate(l1_paths):
            if path.exists():
                df = pd.read_csv(path)
                df["priority"] = idx
                l1_frames.append(df)
        if not l1_frames:
            return pd.DataFrame()
        l1 = pd.concat(l1_frames, ignore_index=True)
        l1 = l1.sort_values("priority").drop_duplicates(subset=["industry_code"], keep="first")

        members["in_date"] = pd.to_datetime(members["in_date"], errors="coerce")
        l2_map = l2[["index_code", "parent_code", "industry_name"]].rename(columns={"industry_name": "industry_name_l2"})
        l1_map = l1[["industry_code", "industry_name"]].rename(
            columns={"industry_code": "parent_code", "industry_name": "industry_name_l1"}
        )

        merged = members.merge(l2_map[["index_code", "parent_code"]], on="index_code", how="left")

        if "industry_name_l2" in merged.columns:
            name_map = l2_map[["industry_name_l2", "parent_code"]].dropna().drop_duplicates(subset=["industry_name_l2"])
            merged = merged.merge(name_map, on="industry_name_l2", how="left", suffixes=("", "_by_name"))
            merged["parent_code"] = merged["parent_code"].fillna(merged["parent_code_by_name"])
            merged = merged.drop(columns=["parent_code_by_name"])

        merged = merged.merge(l1_map, on="parent_code", how="left")

        l1_name_set = set(l1_map["industry_name_l1"].dropna().unique())
        merged["sw_industry"] = merged.get("industry_name_l1")
        if "industry_name_l2" in merged.columns:
            mask = merged["sw_industry"].isna() & merged["industry_name_l2"].isin(l1_name_set)
            merged.loc[mask, "sw_industry"] = merged.loc[mask, "industry_name_l2"]

        merged = merged.sort_values("in_date")
        def _pick_latest(group: pd.DataFrame) -> pd.Series:
            valid = group[group["sw_industry"].notna()]
            if not valid.empty:
                return valid.iloc[-1]
            return group.iloc[-1]

        latest = merged.groupby("con_code", as_index=False).apply(_pick_latest).reset_index(drop=True)
        latest = latest.dropna(subset=["sw_industry"])
        return latest.rename(columns={"con_code": "ts_code"})[["ts_code", "sw_industry"]]

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

        if sw_list:
            filtered = df[df['name'].isin(sw_list)].copy()
            if not filtered.empty:
                df = filtered

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
        if industry_df is None:
            industry_df = pd.DataFrame()

        valuation_df = self.load_sw_valuation_features(sw_list)
        turnover_df = self.load_sw_turnover_rate(sw_list)
        stock_map = self.load_sw_stock_industry_map()
        rev_yoy_df = self.load_industry_rev_yoy(stock_map)

        if industry_df.empty:
            frames = [valuation_df, turnover_df, rev_yoy_df]
            frames = [f for f in frames if f is not None and not f.empty]
            if not frames:
                return industry_df
            base = frames[0][['sw_industry', 'date']].drop_duplicates()
            for frame in frames[1:]:
                base = base.merge(frame[['sw_industry', 'date']].drop_duplicates(),
                                  on=['sw_industry', 'date'], how='outer')
            result = base
        else:
            base = industry_df[['sw_industry', 'date']].drop_duplicates()
            if not rev_yoy_df.empty:
                base = base.merge(
                    rev_yoy_df[['sw_industry', 'date']].drop_duplicates(),
                    on=['sw_industry', 'date'],
                    how='outer'
                )
            result = base.merge(industry_df, on=['sw_industry', 'date'], how='left')

        if not valuation_df.empty:
            result = result.merge(valuation_df, on=['sw_industry', 'date'], how='left')
        if not turnover_df.empty:
            result = result.merge(turnover_df, on=['sw_industry', 'date'], how='left')
        if not rev_yoy_df.empty:
            result = result.merge(rev_yoy_df, on=['sw_industry', 'date'], how='left')

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
            ffill_cols = ["rev_yoy", "turnover_rate", "pb_percentile", "pe_percentile", "rps_120"]
            for col in ffill_cols:
                if col in industry_with_market.columns:
                    industry_with_market[col] = industry_with_market.groupby("sw_industry")[col].ffill()

        # 4. 补齐字段
        industry_final = industry_with_market.copy()
        if 'sw_ppi_yoy' not in industry_final.columns and 'ppi_yoy' in industry_final.columns:
            industry_final['sw_ppi_yoy'] = industry_final['ppi_yoy']

        if 'inventory_yoy' not in industry_final.columns:
            industry_final['inventory_yoy'] = np.nan
        if 'inventory_yoy_source' not in industry_final.columns:
            industry_final['inventory_yoy_source'] = pd.Series([None] * len(industry_final), dtype="object")

        # 主口径：库存同比 = 产量同比 - 营收同比
        if 'output_yoy' in industry_final.columns and 'rev_yoy' in industry_final.columns:
            mask = industry_final['inventory_yoy'].isna() & industry_final['output_yoy'].notna() & industry_final['rev_yoy'].notna()
            industry_final.loc[mask, 'inventory_yoy'] = industry_final.loc[mask, 'output_yoy'] - industry_final.loc[mask, 'rev_yoy']
            industry_final.loc[mask, 'inventory_yoy_source'] = 'calc_output_minus_rev'

        # 备用口径：库存同比 = 产量同比 - 行业PPI同比
        proxy_col = None
        if 'sw_ppi_yoy' in industry_final.columns:
            proxy_col = 'sw_ppi_yoy'
        elif 'ppi_yoy' in industry_final.columns:
            proxy_col = 'ppi_yoy'

        if proxy_col and 'output_yoy' in industry_final.columns:
            mask = industry_final['inventory_yoy'].isna() & industry_final['output_yoy'].notna() & industry_final[proxy_col].notna()
            industry_final.loc[mask, 'inventory_yoy'] = industry_final.loc[mask, 'output_yoy'] - industry_final.loc[mask, proxy_col]
            industry_final.loc[mask, 'inventory_yoy_source'] = f'proxy_output_minus_{proxy_col}'

        # 前向填充并标记来源
        if 'inventory_yoy' in industry_final.columns and 'output_yoy' in industry_final.columns:
            before_ffill = industry_final['inventory_yoy'].copy()
            industry_final['inventory_yoy'] = industry_final.groupby("sw_industry")['inventory_yoy'].ffill()
            filled_mask = before_ffill.isna() & industry_final['inventory_yoy'].notna() & industry_final['output_yoy'].notna()
            industry_final.loc[filled_mask, 'inventory_yoy_source'] = 'ffill'

            # 对于没有output_yoy的行，明确不计算inventory_yoy
            no_output_mask = industry_final['output_yoy'].isna()
            industry_final.loc[no_output_mask, 'inventory_yoy'] = np.nan
            industry_final.loc[no_output_mask, 'inventory_yoy_source'] = None

        # 标记行业口径：非工业行业直接标记为N/A
        non_industrial_list = {
            "银行", "非银金融", "房地产", "交通运输", "公用事业",
            "商贸零售", "社会服务", "传媒", "通信", "计算机",
            "综合", "美容护理", "农林牧渔", "建筑装饰", "医药生物"
        }
        industry_final["industry_scope"] = "industrial"
        industry_final.loc[industry_final["sw_industry"].isin(non_industrial_list), "industry_scope"] = "non_industrial"

        # 非工业行业：明确不计算工业口径字段
        non_ind_mask = industry_final["industry_scope"] == "non_industrial"
        for col in ["output_yoy", "sw_ppi_yoy", "fai_yoy", "inventory_yoy"]:
            if col in industry_final.columns:
                industry_final.loc[non_ind_mask, col] = np.nan
        if "inventory_yoy_source" in industry_final.columns:
            industry_final.loc[non_ind_mask, "inventory_yoy_source"] = None

        required_cols = [
            'sw_industry', 'date', 'sw_ppi_yoy', 'fai_yoy',
            'pb_percentile', 'pe_percentile', 'turnover_rate', 'rps_120',
            'inventory_yoy', 'rev_yoy', 'inventory_yoy_source', 'industry_scope'
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

    industry_required = [
        'sw_industry', 'date', 'sw_ppi_yoy', 'fai_yoy',
        'pb_percentile', 'pe_percentile', 'turnover_rate', 'rps_120',
        'inventory_yoy', 'rev_yoy', 'inventory_yoy_source', 'industry_scope'
    ]
    industry_report = _coverage(data['industry'], industry_required)

    industrial_only_cols = ['sw_ppi_yoy', 'fai_yoy', 'output_yoy', 'inventory_yoy']
    if 'industry_scope' in data['industry'].columns:
        industrial_mask = data['industry']['industry_scope'] == 'industrial'
        eligible_null = {}
        for col in industrial_only_cols:
            if col in data['industry'].columns:
                eligible_null[col] = float(data['industry'].loc[industrial_mask, col].isna().mean()) if industrial_mask.any() else 1.0
        industry_report["eligible_ratio"] = float(industrial_mask.mean())
        industry_report["eligible_null_ratio"] = eligible_null

    report = {
        "macro": _coverage(data['macro'], ['date', 'credit_growth', 'pmi', 'cpi_yoy', 'yield_10y']),
        "industry": industry_report,
        "northbound": _coverage(data['northbound'], ['sw_industry', 'trade_date', 'northbound_signal', 'industry_ratio']),
        "causes": {
            "macro": {
                "credit_growth": "社融存量同比口径，前12个月无法计算。",
                "pmi": "依赖tushare_pmi.parquet，缺失或未同步到raw目录。",
                "cpi_yoy": "依赖tushare_cpi.parquet，缺失或未同步到raw目录。",
                "yield_10y": "依赖tushare_yield_10y.parquet或yield_10y.parquet，早期日期可能为空。"
            },
            "industry": {
            "sw_ppi_yoy": "仅工业行业口径；非工业行业直接标记为N/A。工业行业PPI由NBS环比指数链式构建，同比需12个月窗口，早期月份为空。",
            "fai_yoy": "仅工业行业口径；非工业行业直接标记为N/A。",
            "pb_percentile": "依赖sw_valuation行业估值，早期或行业缺口；rev_yoy季度行扩展会产生额外空值。",
            "pe_percentile": "依赖sw_valuation行业估值，早期或行业缺口；rev_yoy季度行扩展会产生额外空值。",
            "turnover_rate": "依赖sw_daily_all的amount/float_mv，缺口或行业未覆盖；rev_yoy季度行扩展会产生额外空值。",
            "rps_120": "需要120个交易日收益率，样本初期为空；rev_yoy季度行扩展会产生额外空值。",
            "rev_yoy": "依赖fina_indicator并映射到行业，财报缺口或映射覆盖不足。",
            "inventory_yoy": "仅在output_yoy存在时计算；主口径为output_yoy - rev_yoy，缺失时备用口径为output_yoy - sw_ppi_yoy/ppi_yoy。"
            },
            "northbound": {
                "northbound_signal": "依赖northbound行业流数据，若行业或日期缺口则为空。",
                "industry_ratio": "来自北向行业占比，原始vol/ratio缺口会导致缺失。"
            }
        }
    }
    report_path = os.path.join(output_dir, 'macro_alignment_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n对齐审计报告已保存: {report_path}")
    
    print(f"\n数据已保存到 {output_dir}")


if __name__ == '__main__':
    main()
