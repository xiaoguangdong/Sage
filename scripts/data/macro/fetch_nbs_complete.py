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

import requests
import pandas as pd
import tushare as ts
import time
import os
from datetime import datetime, timedelta

from tushare_auth import get_tushare_token

class CompleteNBSDataFetcher:
    def __init__(self, tushare_token=None):
        self.base_url = "https://data.stats.gov.cn/easyquery.htm"
        self.output_dir = 'data/tushare/macro'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tushare用于获取PMI数据
        self.tushare_token = get_tushare_token(tushare_token)
        self.pro = ts.pro_api(self.tushare_token)
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
            "Referer": "https://data.stats.gov.cn/easyquery.htm?cn=C01"
        }
        self.session = requests.session()
        self.api_delay = 3  # NBS API请求间隔3秒
    
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
            "h": 1
        }

        print(f"[{datetime.now().strftime('%H:%M:%S')}] 请求数据：{zb_code}, 时间范围：{sj_code}...")
        response = self.session.get(self.base_url, params=params, headers=self.headers)

        if response.status_code != 200:
            print(f"[DEBUG] 请求失败，状态码: {response.status_code}")
            print(f"[DEBUG] 响应内容: {response.text[:200]}")
            return None

        print(f"[DEBUG] 请求成功，开始解析数据...")
        data = response.json()
        
        # 调试：检查返回的数据结构
        returncode = data.get('returncode', '')
        print(f"[DEBUG] returncode={returncode}")
        
        if returncode != 200:
            print(f"[DEBUG] NBS返回错误: {data.get('returndata', '')}")
            return None
        
        returndata = data.get('returndata', {})
        wdnodes = returndata.get('wdnodes', [])
        datanodes = returndata.get('datanodes', [])
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
    
    def fetch_nbs_cpi_national(self):
        """
        获取全国CPI数据
        """
        print("\n=== 获取全国CPI数据 ===")
        df = self.fetch_from_nbs(
            dbcode="hgyd",
            rowcode="zb",
            colcode="sj",
            zb_code="A0101",  # CPI
            sj_code="LAST36"
        )

        if df is not None:
            # 重命名列并转换格式
            df = df.rename(columns={"名称": "指标"})
            filepath = os.path.join(self.output_dir, "nbs_cpi_national.csv")
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"✓ 全国CPI数据已保存: {filepath} ({len(df)}行)")
            return df
        return None
    
    def fetch_nbs_ppi_national(self):
        """
        获取全国PPI数据
        """
        print("\n=== 获取全国PPI数据 ===")
        df = self.fetch_from_nbs(
            dbcode="hgyd",
            rowcode="zb",
            colcode="sj",
            zb_code="A010203",  # PPI
            sj_code="LAST36"
        )

        if df is not None:
            df = df.rename(columns={"名称": "指标"})
            filepath = os.path.join(self.output_dir, "nbs_ppi_national.csv")
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"✓ 全国PPI数据已保存: {filepath} ({len(df)}行)")
            return df
        return None
    
    def fetch_nbs_ppi_industry(self):
        """
        获取分行业PPI数据（A010F）
        
        NBS代码: A010F - 工业生产者出厂价格指数(上月=100)
        包含41个行业的PPI环比数据
        """
        print("\n=== 获取分行业PPI数据（A010F）===")
        
        df = self.fetch_from_nbs(
            dbcode="hgyd",
            rowcode="zb",
            colcode="sj",
            zb_code="A010F",
            sj_code="2020-2025"  # 获取2020-2025年的数据
        )

        if df is not None and len(df) > 0:
            # 转换为长格式
            df_melted = df.melt(
                id_vars=["名称"],
                var_name="date",
                value_name="ppi_mom"  # 环比
            )

            # 添加行业信息
            df_melted['industry'] = df_melted['名称']
            df_melted['industry_code'] = "A010F"

            # 清理数据
            df_melted = df_melted[['industry', 'industry_code', 'date', 'ppi_mom']]
            
            # 清理日期格式
            df_melted['date'] = df_melted['date'].str.replace('年', '-').str.replace('月', '-01')
            
            # 添加年份和月份
            df_melted['year'] = pd.to_datetime(df_melted['date']).dt.year
            df_melted['month'] = pd.to_datetime(df_melted['date']).dt.month
            
            # 筛选2020年以后的数据
            df_melted = df_melted[df_melted['year'] >= 2020]
            
            # 转换数值
            df_melted['ppi_mom'] = pd.to_numeric(df_melted['ppi_mom'], errors='coerce')
            
            # 排序
            df_melted = df_melted.sort_values(['industry', 'date'])

            # 保存
            filepath = os.path.join(self.output_dir, "nbs_ppi_industry_2020.csv")
            df_melted.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"\n✓ 分行业PPI数据已保存: {filepath}")
            print(f"  总行数: {len(df_melted)}")
            print(f"  行业数: {df_melted['industry'].nunique()}")
            print(f"  时间范围: {df_melted['date'].min()} 至 {df_melted['date'].max()}")

            return df_melted
        return None
    
    def fetch_nbs_fai_industry(self):
        """
        获取分行业固定资产投资数据（A0403）
        
        NBS代码: A0403 - 固定资产投资额_累计增长
        包含74个行业的固定资产投资同比增长数据
        """
        print("\n=== 获取分行业固定资产投资数据（A0403）===")
        
        df = self.fetch_from_nbs(
            dbcode="hgyd",
            rowcode="zb",
            colcode="sj",
            zb_code="A0403",
            sj_code="2020-2025"  # 获取2020-2025年的数据
        )

        if df is not None and len(df) > 0:
            # 转换为长格式
            df_melted = df.melt(
                id_vars=["名称"],
                var_name="date",
                value_name="fai_yoy"  # 固定资产投资同比
            )

            # 添加行业信息
            df_melted['industry'] = df_melted['名称']
            df_melted['industry_code'] = "A0403"

            # 清理数据
            df_melted = df_melted[['industry', 'industry_code', 'date', 'fai_yoy']]
            
            # 清理日期格式
            df_melted['date'] = df_melted['date'].str.replace('年', '-').str.replace('月', '-01')
            
            # 添加年份和月份
            df_melted['year'] = pd.to_datetime(df_melted['date']).dt.year
            df_melted['month'] = pd.to_datetime(df_melted['date']).dt.month
            
            # 筛选2020年以后的数据
            df_melted = df_melted[df_melted['year'] >= 2020]
            
            # 转换数值
            df_melted['fai_yoy'] = pd.to_numeric(df_melted['fai_yoy'], errors='coerce')
            
            # 排序
            df_melted = df_melted.sort_values(['industry', 'date'])

            # 保存
            filepath = os.path.join(self.output_dir, "nbs_fai_industry_2020.csv")
            df_melted.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"\n✓ 固定资产投资数据已保存: {filepath}")
            print(f"  总行数: {len(df_melted)}")
            print(f"  行业数: {df_melted['industry'].nunique()}")
            print(f"  时间范围: {df_melted['date'].min()} 至 {df_melted['date'].max()}")

            return df_melted
        return None
    
    def fetch_nbs_output_products(self):
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
            
            df = self.fetch_from_nbs(
                dbcode="hgyd",
                rowcode="zb",
                colcode="sj",
                zb_code=product_code,
                sj_code="2020-2025"  # 获取2020-2025年的数据
            )

            if df is not None and len(df) > 0:
                # 提取产品名称
                product_name = df.iloc[0, 0] if len(df) > 0 else product_code
                
                # 转换为长格式
                df_melted = df.melt(
                    id_vars=["名称"],
                    var_name="date",
                    value_name="output_value"  # 产量值
                )

                # 添加产品信息
                df_melted['product'] = product_name
                df_melted['product_code'] = product_code

                # 清理数据
                df_melted = df_melted[['product', 'product_code', 'date', 'output_value']]
                all_data.append(df_melted)
                
                print(f"OK ({len(df_melted)}行)")
            else:
                print("失败")
            
            # 请求间隔
            time.sleep(self.api_delay)

        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            
            # 清理日期格式
            result_df['date'] = result_df['date'].str.replace('年', '-').str.replace('月', '-01')
            
            # 添加年份和月份
            result_df['year'] = pd.to_datetime(result_df['date']).dt.year
            result_df['month'] = pd.to_datetime(result_df['date']).dt.month
            
            # 筛选2020年以后的数据
            result_df = result_df[result_df['year'] >= 2020]
            
            # 转换数值
            result_df['output_value'] = pd.to_numeric(result_df['output_value'], errors='coerce')
            
            # 排序
            result_df = result_df.sort_values(['product', 'date'])

            # 保存
            filepath = os.path.join(self.output_dir, "nbs_output_2020.csv")
            result_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"\n✓ 工业品产量数据已保存: {filepath}")
            print(f"  总行数: {len(result_df)}")
            print(f"  产品数: {result_df['product'].nunique()}")
            print(f"  时间范围: {result_df['date'].min()} 至 {result_df['date'].max()}")

            return result_df
        return None
    
    def fetch_tushare_pmi(self, start_m='202001', end_m='202512'):
        """
        从Tushare获取PMI数据

        Args:
            start_m: 开始月份（YYYYMM格式）
            end_m: 结束月份（YYYYMM格式）
        """
        print("\n=== 从Tushare获取PMI数据 ===")

        try:
            df = self.pro.cn_pmi(start_m=start_m, end_m=end_m)

            if df is not None and len(df) > 0:
                # 保存为Parquet格式
                filepath = os.path.join(self.output_dir, "tushare_pmi.parquet")
                df.to_parquet(filepath, index=False)
                print(f"✓ PMI数据已保存: {filepath} ({len(df)}行)")

                return df
        except Exception as e:
            print(f"获取PMI数据失败: {e}")

        return None
    
    def fetch_tushare_cpi(self, start_m='202001', end_m='202512'):
        """
        从Tushare获取CPI数据

        Args:
            start_m: 开始月份（YYYYMM格式）
            end_m: 结束月份（YYYYMM格式）
        """
        print("\n=== 从Tushare获取CPI数据 ===")

        try:
            df = self.pro.cn_cpi(start_m=start_m, end_m=end_m)

            if df is not None and len(df) > 0:
                # 保存为Parquet格式
                filepath = os.path.join(self.output_dir, "tushare_cpi.parquet")
                df.to_parquet(filepath, index=False)
                print(f"✓ CPI数据已保存: {filepath} ({len(df)}行)")

                return df
        except Exception as e:
            print(f"获取CPI数据失败: {e}")

        return None
    
    def fetch_tushare_ppi(self, start_m='202001', end_m='202512'):
        """
        从Tushare获取PPI数据

        Args:
            start_m: 开始月份（YYYYMM格式）
            end_m: 结束月份（YYYYMM格式）
        """
        print("\n=== 从Tushare获取PPI数据 ===")

        try:
            df = self.pro.cn_ppi(start_m=start_m, end_m=end_m)

            if df is not None and len(df) > 0:
                # 保存为Parquet格式
                filepath = os.path.join(self.output_dir, "tushare_ppi.parquet")
                df.to_parquet(filepath, index=False)
                print(f"✓ PPI数据已保存: {filepath} ({len(df)}行)")

                return df
        except Exception as e:
            print(f"获取PPI数据失败: {e}")

        return None
    
    def fetch_all(self, start_date='20200101', end_date='20251231'):
        """
        获取所有NBS数据

        Args:
            start_date: 开始日期（YYYYMMDD格式）
            end_date: 结束日期（YYYYMMDD格式）
        """
        print("=" * 80)
        print("开始获取完整的NBS宏观数据")
        print(f"时间范围: 2020年至今")
        print("=" * 80)

        start_m = start_date[:6]
        end_m = end_date[:6]

        # 1. 从NBS获取数据
        print("\n【步骤1】从NBS获取数据")
        print("-" * 80)

        # 1.1 全国CPI
        self.fetch_nbs_cpi_national()

        # 1.2 全国PPI
        self.fetch_nbs_ppi_national()

        # 1.3 分行业PPI（A010F - 41个行业）
        self.fetch_nbs_ppi_industry()

        # 1.4 分行业固定资产投资（A0403 - 74个行业）
        self.fetch_nbs_fai_industry()

        # 1.5 工业品产量（A020901-A020929 - 29个产品）
        self.fetch_nbs_output_products()

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
    fetcher = CompleteNBSDataFetcher()
    fetcher.fetch_all(start_date='20200101', end_date='20251231')


if __name__ == '__main__':
    main()
