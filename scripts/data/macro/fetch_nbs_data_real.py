#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
国家统计局真实数据获取脚本

功能：
1. 获取国家统计局API数据
2. 支持CPI、PPI、PMI等指标
3. 支持分行业数据
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime

from scripts.data.macro.paths import MACRO_DIR


class NBSDataFetcher:
    def __init__(self):
        self.base_url = "https://data.stats.gov.cn/easyquery.htm"
        self.output_dir = str(MACRO_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
            "Referer": "https://data.stats.gov.cn/easyquery.htm?cn=C01"
        }
        self.session = requests.session()
        self.api_delay = 3  # NBS API请求间隔3秒
    
    def fetch_data(self, dbcode, rowcode, colcode, zb_code, sj_code=None):
        """
        获取数据
        
        Args:
            dbcode: 数据库代码（月度：hgyd/hgjd/hgnd）
            rowcode: 行代码（zb）
            colcode: 列代码（sj）
            zb_code: 指标代码
            sj_code: 时间代码（可选）
        
        Returns:
            DataFrame: 数据
        """
        # 第一次请求：获取指标数据
        params = {
            "m": "QueryData",
            "dbcode": dbcode,
            "rowcode": rowcode,
            "colcode": colcode,
            "wds": "[]",
            "dfwds": f'[{{"wdcode":"zb","valuecode":"{zb_code}"}}]',
            "k1": str(int(round(time.time() * 1000))),
            "h": 1
        }
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 第一次请求：获取指标数据...")
        response = self.session.get(self.base_url, params=params, headers=self.headers)
        
        if response.status_code != 200:
            print(f"请求失败，状态码: {response.status_code}")
            return None
        
        # 第二次请求：获取时间数据
        if sj_code:
            params['dfwds'] = f'[{{"wdcode":"sj","valuecode":"{sj_code}"}}]'
        else:
            params['dfwds'] = '[{"wdcode":"sj","valuecode":"LAST60"}]'  # 最近60个月
        
        params['k1'] = str(int(round(time.time() * 1000)))
        del params['h']
        
        time.sleep(self.api_delay)  # 两次请求之间等待3秒
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 第二次请求：获取时间数据...")
        response = self.session.get(self.base_url, params=params, headers=self.headers)
        
        if response.status_code != 200:
            print(f"请求失败，状态码: {response.status_code}")
            return None
        
        data = response.json()
        return self._parse_data(data)
    
    def _parse_data(self, data):
        """
        解析返回的JSON数据
        """
        wdnodes = data.get("returndata", {}).get("wdnodes", [])
        
        if len(wdnodes) < 2:
            print("数据格式错误：wdnodes长度不足")
            return None
        
        # 提取行名称和列头
        first_col = [node.get("cname", "") for node in wdnodes[0].get("nodes", [])]
        col_headers = [node.get("cname", "") for node in wdnodes[1].get("nodes", [])]
        
        # 提取数据
        datanodes = data.get("returndata", {}).get("datanodes", [])
        
        rows = len(first_col)
        cols = len(col_headers)
        
        # 创建结果矩阵
        result = [["" for _ in range(cols + 1)] for _ in range(rows)]
        
        # 填充第一列（名称）
        for i in range(rows):
            result[i][0] = first_col[i]
        
        # 填充数据列
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
                    value = round(float(value)) if value != "" else ""
                except ValueError:
                    value = ""
                result[row_index][col_index + 1] = value
        
        # 转换为DataFrame
        df = pd.DataFrame(result, columns=["名称"] + col_headers)
        return df
    
    def save_data(self, df, filename):
        """
        保存数据
        """
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 数据已保存: {filename}")
    
    def test_fetch(self):
        """
        测试获取数据
        """
        print("=== 测试获取国家统计局数据 ===\n")
        
        # 测试1：获取CPI数据（全国）
        print("测试1：获取CPI数据（全国）...")
        cpi_df = self.fetch_data(
            dbcode="hgyd",  # 月度数据
            rowcode="zb",
            colcode="sj",
            zb_code="A0101",  # CPI
            sj_code="LAST12"
        )
        
        if cpi_df is not None:
            print(f"成功获取CPI数据: {len(cpi_df)}行 × {len(cpi_df.columns)}列")
            print(f"列名: {cpi_df.columns.tolist()}")
            print(f"前3行:")
            print(cpi_df.head(3))
            self.save_data(cpi_df, "nbs_cpi_national.csv")
        
        # 测试2：获取PPI数据（全国）
        print(f"\n测试2：获取PPI数据（全国）...")
        ppi_df = self.fetch_data(
            dbcode="hgyd",
            rowcode="zb",
            colcode="sj",
            zb_code="A010203",  # PPI
            sj_code="LAST12"
        )
        
        if ppi_df is not None:
            print(f"成功获取PPI数据: {len(ppi_df)}行 × {len(ppi_df.columns)}列")
            print(f"列名: {ppi_df.columns.tolist()}")
            print(f"前3行:")
            print(ppi_df.head(3))
            self.save_data(ppi_df, "nbs_ppi_national.csv")


def main():
    fetcher = NBSDataFetcher()
    fetcher.test_fetch()


if __name__ == '__main__':
    main()
