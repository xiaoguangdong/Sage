#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
获取完整的NBS（国家统计局）数据

根据《宏观经济预测模型设计文档》要求，需要获取：
1. 分行业PPI（32个工业行业）
2. 分行业固定资产投资
3. 主要工业品产量
4. 全国CPI
5. 全国PPI
6. PMI（从Tushare获取）

数据来源：
- 国家统计局官网：https://data.stats.gov.cn/easyquery.htm
- Tushare API：PMI数据
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.macro.paths import MACRO_DIR


class CompleteNBSDataFetcher:
    def __init__(self, tushare_token=None):
        self.base_url = "https://data.stats.gov.cn/easyquery.htm"
        self.output_dir = str(MACRO_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
            "Referer": "https://data.stats.gov.cn/easyquery.htm?cn=C01",
        }
        self.session = requests.session()
        self.api_delay = 3  # NBS API请求间隔3秒

    @staticmethod
    def _build_sj_ranges(start_year: Optional[int], end_year: Optional[int], span_years: int = 5) -> List[str]:
        if not start_year or not end_year:
            return []
        if start_year > end_year:
            start_year, end_year = end_year, start_year
        ranges = []
        year = start_year
        while year <= end_year:
            end = min(end_year, year + span_years - 1)
            ranges.append(f"{year}-{end}")
            year = end + 1
        return ranges

    @staticmethod
    def _merge_wide_frames(frames: Iterable[pd.DataFrame], key: str = "名称") -> Optional[pd.DataFrame]:
        merged: Optional[pd.DataFrame] = None
        for df in frames:
            if df is None or df.empty:
                continue
            if merged is None:
                merged = df.copy()
                continue
            merged = merged.merge(df, on=key, how="outer", suffixes=("", "_dup"))
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            for col in dup_cols:
                base = col[: -len("_dup")]
                if base in merged.columns:
                    base_series = merged[base].replace("", pd.NA)
                    dup_series = merged[col].replace("", pd.NA)
                    merged[base] = base_series.fillna(dup_series)
                    merged.drop(columns=[col], inplace=True)
                else:
                    merged.rename(columns={col: base}, inplace=True)
            merged = merged.loc[:, ~merged.columns.duplicated()]
        return merged

    def fetch_from_nbs(self, dbcode, rowcode, colcode, zb_code, sj_code="LAST24"):
        """
        从NBS获取数据

        Args:
            dbcode: 数据库代码（hgyd=月度, hgjd=季度, hgnd=年度）
            rowcode: 行代码（zb）
            colcode: 列代码（sj）
            zb_code: 指标代码
            sj_code: 时间代码（LAST24=最近24个月，或"2020-2025"表示时间范围）

        Returns:
            DataFrame: 数据
        """
        print(f"[DEBUG] fetch_from_nbs() 开始 - zb_code={zb_code}, sj_code={sj_code}")

        # 直接使用指定的指标和时间参数
        params = {
            "m": "QueryData",
            "dbcode": dbcode,
            "rowcode": rowcode,
            "colcode": colcode,
            "wds": "[]",
            "dfwds": f'[{{"wdcode":"zb","valuecode":"{zb_code}"}}{{"wdcode":"sj","valuecode":"{sj_code}"}}]',
            "k1": str(int(round(time.time() * 1000))),
            "h": 1,
        }

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 请求数据：{zb_code}, 时间范围：{sj_code}...")
        response = self.session.get(self.base_url, params=params, headers=self.headers)

        if response.status_code != 200:
            print(f"[DEBUG] 请求失败，状态码: {response.status_code}")
            print(f"[DEBUG] 响应内容: {response.text[:200]}")
            return None

        print("[DEBUG] 请求成功，开始解析数据...")
        data = response.json()

        # 调试：检查返回的数据结构
        returncode = data.get("returncode", "")
        print(f"[DEBUG] returncode={returncode}")

        if returncode != 200:
            print(f"[DEBUG] NBS返回错误: {data.get('returndata', '')}")
            return None

        returndata = data.get("returndata", {})
        wdnodes = returndata.get("wdnodes", [])
        datanodes = returndata.get("datanodes", [])
        print(f"[DEBUG] wdnodes数量={len(wdnodes)}, datanodes数量={len(datanodes)}")

        result = self._parse_nbs_data(data)
        print(f"[DEBUG] 解析完成，返回 {len(result)}行 x {len(result.columns)}列")
        return result

    def _parse_nbs_data(self, data):
        """
        解析NBS返回的JSON数据
        """
        print("[DEBUG] _parse_nbs_data() 开始")

        wdnodes = data.get("returndata", {}).get("wdnodes", [])
        print(f"[DEBUG] wdnodes长度: {len(wdnodes)}")

        if len(wdnodes) < 2:
            print("[DEBUG] 数据格式错误：wdnodes长度不足")
            return None

        # 提取行名称和列头
        first_col = [node.get("cname", "") for node in wdnodes[0].get("nodes", [])]
        col_headers = [node.get("cname", "") for node in wdnodes[1].get("nodes", [])]

        print(f"[DEBUG] 行数据数: {len(first_col)}, 列数据数: {len(col_headers)}")
        print(f"[DEBUG] 前3个指标名称: {first_col[:3]}")
        print(f"[DEBUG] 前3个时间列: {col_headers[:3]}")

        # 提取数据
        datanodes = data.get("returndata", {}).get("datanodes", [])
        print(f"[DEBUG] 数据点数量: {len(datanodes)}")

        rows = len(first_col)
        cols = len(col_headers)

        # 创建结果矩阵
        result = [["" for _ in range(cols + 1)] for _ in range(rows)]

        # 填充第一列（名称）
        for i in range(rows):
            result[i][0] = first_col[i]

        # 填充数据列
        filled_count = 0
        for node in datanodes:
            row_code = node.get("wds", [])[0].get("valuecode", "")
            col_code = node.get("wds", [])[1].get("valuecode", "")
            value = node.get("data", {}).get("data", "")

            # 找到对应的行列索引
            row_index = next((i for i, n in enumerate(wdnodes[0]["nodes"]) if n.get("code") == row_code), None)
            col_index = next((j for j, n in enumerate(wdnodes[1]["nodes"]) if n.get("code") == col_code), None)

            if row_index is not None and col_index is not None:
                # 尝试转换为数值
                try:
                    value = round(float(value), 1) if value != "" else ""
                except ValueError:
                    value = ""
                result[row_index][col_index + 1] = value
                filled_count += 1

        print(f"[DEBUG] 成功填充 {filled_count} 个数据点")

        # 转换为DataFrame
        df = pd.DataFrame(result, columns=["名称"] + col_headers)
        print(f"[DEBUG] DataFrame创建成功: {len(df)}行 x {len(df.columns)}列")
        return df

    def fetch_nbs_cpi_national(self, sj_ranges: Optional[List[str]] = None, zb_code: str = "A010101"):
        """
        获取全国CPI数据
        """
        print("\n=== 获取全国CPI数据 ===")
        if sj_ranges:
            dfs = []
            for sj_code in sj_ranges:
                dfs.append(
                    self.fetch_from_nbs(dbcode="hgyd", rowcode="zb", colcode="sj", zb_code=zb_code, sj_code=sj_code)
                )
                time.sleep(self.api_delay)
            df = self._merge_wide_frames(dfs)
        else:
            df = self.fetch_from_nbs(
                dbcode="hgyd", rowcode="zb", colcode="sj", zb_code=zb_code, sj_code="LAST36"
            )  # CPI

        if df is not None:
            # 重命名列并转换格式
            df = df.rename(columns={"名称": "指标"})
            filepath = os.path.join(self.output_dir, "nbs_cpi_national.csv")
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            print(f"✓ 全国CPI数据已保存: {filepath} ({len(df)}行)")
            return df
        return None

    def fetch_nbs_ppi_national(self, sj_ranges: Optional[List[str]] = None):
        """
        获取全国PPI数据
        """
        print("\n=== 获取全国PPI数据 ===")
        if sj_ranges:
            dfs = []
            for sj_code in sj_ranges:
                dfs.append(
                    self.fetch_from_nbs(dbcode="hgyd", rowcode="zb", colcode="sj", zb_code="A010203", sj_code=sj_code)
                )
                time.sleep(self.api_delay)
            df = self._merge_wide_frames(dfs)
        else:
            df = self.fetch_from_nbs(
                dbcode="hgyd", rowcode="zb", colcode="sj", zb_code="A010203", sj_code="LAST36"
            )  # PPI

        if df is not None:
            df = df.rename(columns={"名称": "指标"})
            filepath = os.path.join(self.output_dir, "nbs_ppi_national.csv")
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            print(f"✓ 全国PPI数据已保存: {filepath} ({len(df)}行)")
            return df
        return None

    def fetch_nbs_ppi_industry(self, sj_ranges: Optional[List[str]] = None, start_year: int = 2020):
        """
        获取分行业PPI数据（A010F）

        NBS代码: A010F - 工业生产者出厂价格指数(上月=100)
        包含41个行业的PPI环比数据
        """
        print("\n=== 获取分行业PPI数据（A010F）===")

        dfs = []
        if sj_ranges:
            for sj_code in sj_ranges:
                dfs.append(
                    self.fetch_from_nbs(dbcode="hgyd", rowcode="zb", colcode="sj", zb_code="A010F", sj_code=sj_code)
                )
                time.sleep(self.api_delay)
        else:
            dfs.append(
                self.fetch_from_nbs(dbcode="hgyd", rowcode="zb", colcode="sj", zb_code="A010F", sj_code="2020-2025")
            )

        merged = self._merge_wide_frames(dfs)

        if merged is not None and len(merged) > 0:
            # 转换为长格式
            df_melted = merged.melt(id_vars=["名称"], var_name="date", value_name="ppi_mom")  # 环比

            # 添加行业信息
            df_melted["industry"] = df_melted["名称"]
            df_melted["industry_code"] = "A010F"

            # 清理数据
            df_melted = df_melted[["industry", "industry_code", "date", "ppi_mom"]]

            # 清理日期格式
            df_melted["date"] = df_melted["date"].str.replace("年", "-").str.replace("月", "-01")

            # 添加年份和月份
            df_melted["year"] = pd.to_datetime(df_melted["date"]).dt.year
            df_melted["month"] = pd.to_datetime(df_melted["date"]).dt.month

            # 筛选起始年份
            df_melted = df_melted[df_melted["year"] >= start_year]

            # 转换数值
            df_melted["ppi_mom"] = pd.to_numeric(df_melted["ppi_mom"], errors="coerce")

            # 排序
            df_melted = df_melted.sort_values(["industry", "date"])

            # 保存
            filepath = os.path.join(self.output_dir, f"nbs_ppi_industry_{start_year}.csv")
            df_melted.to_csv(filepath, index=False, encoding="utf-8-sig")
            print(f"\n✓ 分行业PPI数据已保存: {filepath}")
            print(f"  总行数: {len(df_melted)}")
            print(f"  行业数: {df_melted['industry'].nunique()}")
            print(f"  时间范围: {df_melted['date'].min()} 至 {df_melted['date'].max()}")

            return df_melted
        return None

    def fetch_nbs_fai_industry(self, sj_ranges: Optional[List[str]] = None, start_year: int = 2020):
        """
        获取分行业固定资产投资数据（A0403）

        NBS代码: A0403 - 固定资产投资额_累计增长
        包含74个行业的固定资产投资同比增长数据
        """
        print("\n=== 获取分行业固定资产投资数据（A0403）===")

        dfs = []
        if sj_ranges:
            for sj_code in sj_ranges:
                dfs.append(
                    self.fetch_from_nbs(dbcode="hgyd", rowcode="zb", colcode="sj", zb_code="A0403", sj_code=sj_code)
                )
                time.sleep(self.api_delay)
        else:
            dfs.append(
                self.fetch_from_nbs(dbcode="hgyd", rowcode="zb", colcode="sj", zb_code="A0403", sj_code="2020-2025")
            )

        merged = self._merge_wide_frames(dfs)

        if merged is not None and len(merged) > 0:
            # 转换为长格式
            df_melted = merged.melt(id_vars=["名称"], var_name="date", value_name="fai_yoy")  # 固定资产投资同比

            # 添加行业信息
            df_melted["industry"] = df_melted["名称"]
            df_melted["industry_code"] = "A0403"

            # 清理数据
            df_melted = df_melted[["industry", "industry_code", "date", "fai_yoy"]]

            # 清理日期格式
            df_melted["date"] = df_melted["date"].str.replace("年", "-").str.replace("月", "-01")

            # 添加年份和月份
            df_melted["year"] = pd.to_datetime(df_melted["date"]).dt.year
            df_melted["month"] = pd.to_datetime(df_melted["date"]).dt.month

            # 筛选起始年份
            df_melted = df_melted[df_melted["year"] >= start_year]

            # 转换数值
            df_melted["fai_yoy"] = pd.to_numeric(df_melted["fai_yoy"], errors="coerce")

            # 排序
            df_melted = df_melted.sort_values(["industry", "date"])

            # 保存
            filepath = os.path.join(self.output_dir, f"nbs_fai_industry_{start_year}.csv")
            df_melted.to_csv(filepath, index=False, encoding="utf-8-sig")
            print(f"\n✓ 固定资产投资数据已保存: {filepath}")
            print(f"  总行数: {len(df_melted)}")
            print(f"  行业数: {df_melted['industry'].nunique()}")
            print(f"  时间范围: {df_melted['date'].min()} 至 {df_melted['date'].max()}")

            return df_melted
        return None

    def fetch_nbs_output_products(self, sj_ranges: Optional[List[str]] = None, start_year: int = 2020):
        """
        获取工业品产量数据（A020901-A020929）

        NBS代码: A020901-A020929 - 工业品产量_当期值
        包含29种工业产品的产量数据
        """
        print("\n=== 获取工业品产量数据（A020901-A020929） ===")

        all_data = []

        # 29个产品代码（A020901-A020929）
        product_codes = [f"A0209{i:02d}" for i in range(1, 30)]

        for i, product_code in enumerate(product_codes):
            print(f"  [{i+1}/29] 获取 {product_code}...", end=" ")
            dfs = []
            if sj_ranges:
                for sj_code in sj_ranges:
                    dfs.append(
                        self.fetch_from_nbs(
                            dbcode="hgyd",
                            rowcode="zb",
                            colcode="sj",
                            zb_code=product_code,
                            sj_code=sj_code,
                        )
                    )
                    time.sleep(self.api_delay)
            else:
                dfs.append(
                    self.fetch_from_nbs(
                        dbcode="hgyd",
                        rowcode="zb",
                        colcode="sj",
                        zb_code=product_code,
                        sj_code="2020-2025",
                    )
                )

            merged = self._merge_wide_frames(dfs)
            if merged is not None and len(merged) > 0:
                # 提取产品名称
                product_name = merged.iloc[0, 0] if len(merged) > 0 else product_code

                # 转换为长格式
                df_melted = merged.melt(id_vars=["名称"], var_name="date", value_name="output_value")  # 产量值

                # 添加产品信息
                df_melted["product"] = product_name
                df_melted["product_code"] = product_code

                # 清理数据
                df_melted = df_melted[["product", "product_code", "date", "output_value"]]
                all_data.append(df_melted)

                print(f"OK ({len(df_melted)}行)")
            else:
                print("失败")

            # 请求间隔
            time.sleep(self.api_delay)

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)

            # 清理日期格式
            result_df["date"] = result_df["date"].str.replace("年", "-").str.replace("月", "-01")

            # 添加年份和月份
            result_df["year"] = pd.to_datetime(result_df["date"]).dt.year
            result_df["month"] = pd.to_datetime(result_df["date"]).dt.month

            # 筛选起始年份
            result_df = result_df[result_df["year"] >= start_year]

            # 转换数值
            result_df["output_value"] = pd.to_numeric(result_df["output_value"], errors="coerce")

            # 排序
            result_df = result_df.sort_values(["product", "date"])

            # 保存
            filepath = os.path.join(self.output_dir, f"nbs_output_{start_year}.csv")
            result_df.to_csv(filepath, index=False, encoding="utf-8-sig")
            print(f"\n✓ 工业品产量数据已保存: {filepath}")
            print(f"  总行数: {len(result_df)}")
            print(f"  产品数: {result_df['product'].nunique()}")
            print(f"  时间范围: {result_df['date'].min()} 至 {result_df['date'].max()}")

            return result_df
        return None

    def fetch_tushare_pmi(self, start_m="202001", end_m="202512"):
        """
        从Tushare获取PMI数据

        Args:
            start_m: 开始月份（YYYYMM格式）
            end_m: 结束月份（YYYYMM格式）
        """
        print("\n=== 从Tushare获取PMI数据 ===")
        filepath = os.path.join(self.output_dir, "tushare_pmi.parquet")
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        print(
            "✗ PMI数据缺失，请先运行：python3 scripts/data/tushare_downloader.py --task cn_pmi --start-date 202001 --end-date YYYYMM --resume"
        )
        return None

    def fetch_tushare_cpi(self, start_m="202001", end_m="202512"):
        """
        从Tushare获取CPI数据

        Args:
            start_m: 开始月份（YYYYMM格式）
            end_m: 结束月份（YYYYMM格式）
        """
        print("\n=== 从Tushare获取CPI数据 ===")
        filepath = os.path.join(self.output_dir, "tushare_cpi.parquet")
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        print(
            "✗ CPI数据缺失，请先运行：python3 scripts/data/tushare_downloader.py --task cn_cpi --start-date 202001 --end-date YYYYMM --resume"
        )
        return None

    def fetch_tushare_ppi(self, start_m="202001", end_m="202512"):
        """
        从Tushare获取PPI数据

        Args:
            start_m: 开始月份（YYYYMM格式）
            end_m: 结束月份（YYYYMM格式）
        """
        print("\n=== 从Tushare获取PPI数据 ===")
        filepath = os.path.join(self.output_dir, "tushare_ppi.parquet")
        if os.path.exists(filepath):
            return pd.read_parquet(filepath)
        print(
            "✗ PPI数据缺失，请先运行：python3 scripts/data/tushare_downloader.py --task cn_ppi --start-date 202001 --end-date YYYYMM --resume"
        )
        return None

    def fetch_all(
        self,
        start_date="20200101",
        end_date="20251231",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        span_years: int = 5,
        cpi_sj_ranges: Optional[List[str]] = None,
    ):
        """
        获取所有NBS数据

        Args:
            start_date: 开始日期（YYYYMMDD格式）
            end_date: 结束日期（YYYYMMDD格式）
        """
        print("=" * 80)
        print("开始获取完整的NBS宏观数据")
        print("时间范围:", start_date[:4], "至", end_date[:4])
        print("=" * 80)

        start_m = start_date[:6]
        end_m = end_date[:6]

        # 1. 从NBS获取数据
        print("\n【步骤1】从NBS获取数据")
        print("-" * 80)

        sj_ranges = self._build_sj_ranges(start_year, end_year, span_years=span_years)

        # 1.1 全国CPI
        self.fetch_nbs_cpi_national(sj_ranges=cpi_sj_ranges or sj_ranges)

        # 1.2 全国PPI
        self.fetch_nbs_ppi_national(sj_ranges=sj_ranges)

        # 1.3 分行业PPI（A010F - 41个行业）
        self.fetch_nbs_ppi_industry(sj_ranges=sj_ranges, start_year=start_year or 2020)

        # 1.4 分行业固定资产投资（A0403 - 74个行业）
        self.fetch_nbs_fai_industry(sj_ranges=sj_ranges, start_year=start_year or 2020)

        # 1.5 工业品产量（A020901-A020929 - 29个产品）
        self.fetch_nbs_output_products(sj_ranges=sj_ranges, start_year=start_year or 2020)

        # 2. 从Tushare获取数据
        print("\n【步骤2】从Tushare获取数据")
        print("-" * 80)

        # 2.1 PMI
        self.fetch_tushare_pmi(start_m=start_m, end_m=end_m)

        # 2.2 CPI（备用）
        self.fetch_tushare_cpi(start_m=start_m, end_m=end_m)

        # 2.3 PPI（备用）
        self.fetch_tushare_ppi(start_m=start_m, end_m=end_m)

        print("\n" + "=" * 80)
        print("所有NBS数据获取完成！")
        print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="完整NBS数据获取")
    parser.add_argument("--start-year", type=int, default=2016, help="起始年份")
    parser.add_argument("--end-year", type=int, default=datetime.now().year, help="结束年份")
    parser.add_argument("--span-years", type=int, default=5, help="时间分段跨度（年）")
    parser.add_argument("--cpi-range", type=str, default=None, help="仅用于CPI的时间范围（如 2016-2020）")
    parser.add_argument("--only-cpi", action="store_true", help="仅拉取CPI（避免覆盖其它数据）")
    args = parser.parse_args()

    fetcher = CompleteNBSDataFetcher()
    start_year = args.start_year
    end_year = args.end_year
    start_date = f"{start_year}0101"
    end_date = f"{end_year}1231"
    fetcher.api_delay = max(fetcher.api_delay, 3)
    if args.only_cpi:
        fetcher.fetch_nbs_cpi_national(sj_ranges=[args.cpi_range] if args.cpi_range else None)
        return
    fetcher.fetch_all(
        start_date=start_date,
        end_date=end_date,
        start_year=start_year,
        end_year=end_year,
        span_years=args.span_years,
        cpi_sj_ranges=[args.cpi_range] if args.cpi_range else None,
    )


if __name__ == "__main__":
    main()
