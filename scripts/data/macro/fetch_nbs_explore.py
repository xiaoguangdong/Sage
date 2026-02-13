#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
探索NBS数据结构

从父页面 https://data.stats.gov.cn/easyquery.htm?cn=A01 获取所有数据分类，
然后探索各个子类别的数据。
"""

import requests
import pandas as pd
import json
import os
from urllib.parse import unquote

from scripts.data.macro.paths import MACRO_DIR

class NBSDataExplorer:
    def __init__(self):
        self.base_url = "https://data.stats.gov.cn/easyquery.htm"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
            "Referer": "https://data.stats.gov.cn/",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        self.session = requests.session()
        self.output_dir = str(MACRO_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_parent_categories(self):
        """
        获取父页面的所有数据分类

        cn=A01 是父页面，包含所有数据分类
        """
        print("=== 获取父页面数据分类 ===")

        params = {
            "m": "QueryData",
            "dbcode": "hgyd",
            "rowcode": "zb",
            "colcode": "sj",
            "wds": "[]",
            "dfwds": "[]",
            "k1": str(int(pd.Timestamp.now().timestamp() * 1000)),
            "h": 1
        }

        try:
            response = self.session.get(self.base_url, params=params, headers=self.headers)
            print(f"URL: {response.url}")
            print(f"状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                self._save_raw_data(data, 'parent_categories.json')
                self._parse_parent_categories(data)
                return data
            else:
                print(f"请求失败: {response.status_code}")
                return None

        except Exception as e:
            print(f"错误: {e}")
            return None

    def _parse_parent_categories(self, data):
        """
        解析父页面的数据分类
        """
        wdnodes = data.get("returndata", {}).get("wdnodes", [])

        if len(wdnodes) > 0:
            nodes = wdnodes[0].get("nodes", [])
            print(f"\n找到 {len(nodes)} 个数据分类:")
            print("=" * 100)

            categories = []
            for node in nodes:
                code = node.get("code", "")
                name = node.get("cname", "")
                print(f"  {code}: {name}")
                categories.append({"code": code, "name": name})

            # 保存分类列表
            df = pd.DataFrame(categories)
            filepath = os.path.join(self.output_dir, 'nbs_categories.csv')
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"\n✓ 数据分类已保存: {filepath}")

            return categories
        else:
            print("未找到数据分类")
            return []

    def _save_raw_data(self, data, filename):
        """
        保存原始JSON数据
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ 原始数据已保存: {filepath}")

    def fetch_category_data(self, zb_code, zb_name):
        """
        获取指定分类的数据

        Args:
            zb_code: 指标代码
            zb_name: 指标名称（用于显示）
        """
        print(f"\n=== 获取 {zb_name} ({zb_code}) ===")

        params = {
            "m": "QueryData",
            "dbcode": "hgyd",
            "rowcode": "zb",
            "colcode": "sj",
            "wds": "[]",
            "dfwds": f'[{{"wdcode":"zb","valuecode":"{zb_code}"}}]',
            "k1": str(int(pd.Timestamp.now().timestamp() * 1000)),
            "h": 1
        }

        try:
            response = self.session.get(self.base_url, params=params, headers=self.headers)

            if response.status_code == 200:
                data = response.json()

                # 保存原始数据
                filename = f'{zb_code}_{zb_name}.json'
                self._save_raw_data(data, filename)

                # 解析数据
                result = self._parse_category_data(data, zb_code, zb_name)

                return result
            else:
                print(f"请求失败: {response.status_code}")
                return None

        except Exception as e:
            print(f"错误: {e}")
            return None

    def _parse_category_data(self, data, zb_code, zb_name):
        """
        解析分类数据
        """
        wdnodes = data.get("returndata", {}).get("wdnodes", [])

        if len(wdnodes) >= 2:
            # 提取行数据（行业/产品）
            row_nodes = wdnodes[0].get("nodes", [])
            col_nodes = wdnodes[1].get("nodes", [])

            # 提取数据节点
            datanodes = data.get("returndata", {}).get("datanodes", [])

            print(f"  行数: {len(row_nodes)}")
            print(f"  列数: {len(col_nodes)}")
            print(f"  数据点数: {len(datanodes)}")

            # 打印前5行
            print("\n  前5行数据:")
            for i, node in enumerate(row_nodes[:5]):
                print(f"    {node.get('code', '')}: {node.get('cname', '')}")

            return {
                "zb_code": zb_code,
                "zb_name": zb_name,
                "rows": len(row_nodes),
                "cols": len(col_nodes),
                "datanodes": len(datanodes)
            }
        else:
            print("数据格式错误")
            return None

    def explore_specific_categories(self):
        """
        探索用户指定的分类
        """
        # 用户提供的指标代码
        categories = [
            {"code": "A0403", "name": "固定资产投资"},
            {"code": "A020901", "name": "工业品产量"},
            {"code": "A010D02", "name": "价格指数"},
        ]

        results = []

        for cat in categories:
            result = self.fetch_category_data(cat["code"], cat["name"])
            if result:
                results.append(result)

        # 打印汇总
        print("\n" + "=" * 100)
        print("探索结果汇总:")
        print("=" * 100)

        for r in results:
            print(f"  {r['zb_name']} ({r['zb_code']}): {r['rows']}行 × {r['cols']}列, {r['datanodes']}个数据点")

        return results


def main():
    explorer = NBSDataExplorer()

    # 1. 先获取父页面的所有分类
    parent_data = explorer.fetch_parent_categories()

    # 2. 探索用户指定的分类
    results = explorer.explore_specific_categories()


if __name__ == '__main__':
    main()
