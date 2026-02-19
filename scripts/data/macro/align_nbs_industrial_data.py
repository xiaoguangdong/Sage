#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NBS工业品数据对齐脚本

功能：
1. 解析NBS原始JSON数据（工业品产量、固定资产投资、价格指数）
2. 提取关键工业品指标
3. 对齐到申万行业分类
4. 输出可用于预测的特征数据
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.macro.paths import MACRO_DIR


class NBSIndustrialDataAligner:
    """NBS工业品数据对齐器"""

    def __init__(self, data_dir: str = None):
        """
        初始化对齐器

        Args:
            data_dir: 数据目录
        """
        self.data_dir = Path(data_dir) if data_dir else Path(MACRO_DIR)
        self._mapping_cache = None
        self._nbs_industry_cache = None

        # 工业品到申万行业的映射（简化版，需要完善）
        self.product_to_sw_industry = {
            # 钢铁相关
            "A02090101": "钢铁",  # 铁矿石原矿产量
            "A02090105": "钢铁",  # 生铁产量
            "A02090109": "钢铁",  # 粗钢产量
            "A02090113": "钢铁",  # 钢材产量
            # 有色金属相关
            "A02090117": "有色金属",  # 十种有色金属产量
            "A02090121": "有色金属",  # 精炼铜产量
            "A02090125": "有色金属",  # 电解铝产量
            "A02090129": "有色金属",  # 铅产量
            "A02090133": "有色金属",  # 锌产量
            # 化工相关
            "A02090137": "基础化工",  # 硫酸产量
            "A02090141": "基础化工",  # 烧碱产量
            "A02090145": "基础化工",  # 纯碱产量
            "A02090149": "基础化工",  # 乙烯产量
            "A02090153": "基础化工",  # 化肥产量
            "A02090157": "基础化工",  # 化学农药原药产量
            # 建材相关
            "A02090161": "建筑材料",  # 水泥产量
            "A02090165": "建筑材料",  # 平板玻璃产量
            # 能源相关
            "A02090169": "煤炭",  # 原煤产量
            "A02090173": "石油石化",  # 原油产量
            "A02090177": "石油石化",  # 天然气产量
            "A02090181": "电力设备",  # 发电量
            # 汽车相关
            "A02090185": "汽车",  # 汽车产量
            "A02090189": "汽车",  # 轿车产量
            # 电子相关
            "A02090193": "电子",  # 移动通信手持机产量
            "A02090197": "电子",  # 微型计算机设备产量
            "A02090201": "电子",  # 集成电路产量
            # 家电相关
            "A02090205": "家用电器",  # 家用电冰箱产量
            "A02090209": "家用电器",  # 房间空气调节器产量
            "A02090213": "家用电器",  # 家用洗衣机产量
        }

        # 基于产品名称的行业关键词映射（用于 nbs_output_2020.csv 等）
        self.product_keywords = [
            (["船舶", "航空", "航天", "飞机", "火箭", "导弹"], "国防军工"),
            (["动车组", "铁路机车"], "机械设备"),
            (["钢", "铁矿", "生铁", "粗钢", "钢材", "钢筋", "线材", "钢带", "冷轧"], "钢铁"),
            (["玻璃"], "建筑材料"),
            (["汽车", "轿车", "SUV", "载货汽车"], "汽车"),
            (["工业机器人", "拖拉机"], "机械设备"),
            (["大气污染防治设备"], "环保"),
            (["原盐", "磷矿石"], "基础化工"),
            (["乳制品", "成品糖", "白酒", "植物油", "饲料", "鲜、冷藏肉"], "食品饮料"),
        ]
        self.nbs_industry_aliases = {
            "石油开采": "石油和天然气开采业",
            "天然气开采": "石油和天然气开采业",
            "煤炭开采": "煤炭开采和洗选业",
        }

    def _normalize_product_name(self, name: str) -> str:
        if not isinstance(name, str):
            return ""
        value = name.strip()
        for suffix in ["_当期值", "当期值", "本月值"]:
            if value.endswith(suffix):
                value = value[: -len(suffix)]
        value = value.replace("（", "(").replace("）", ")").replace(" ", "")
        return value

    def _normalize_nbs_indicator_name(self, name: str) -> str:
        if not isinstance(name, str):
            return ""
        value = name.strip()
        value = value.replace("（", "(").replace("）", ")").replace(" ", "")
        value = re.sub(r"\(.*?\)", "", value)
        for marker in [
            "固定资产投资额",
            "工业生产者出厂价格指数",
            "工业生产者购进价格指数",
        ]:
            if marker in value:
                value = value.split(marker)[0]
        value = re.sub(r"(累计增长|同比增长|累计|增长|指数).*?$", "", value)
        value = re.sub(r"[_-].*$", "", value)
        return value.strip()

    def _load_nbs_industry_mapping(self):
        if self._nbs_industry_cache is not None:
            return self._nbs_industry_cache

        mapping_path = PROJECT_ROOT / "config" / "sw_nbs_mapping.yaml"
        nbs_to_sw = {}
        nbs_weights = {}
        nbs_names = []

        if mapping_path.exists():
            try:
                import yaml  # type: ignore

                cfg = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
                sw_to_nbs = cfg.get("sw_to_nbs") or {}
                for sw_name, items in sw_to_nbs.items():
                    if not items:
                        continue
                    for item in items:
                        nbs_name = (item or {}).get("nbs_industry")
                        weight = float((item or {}).get("weight", 0) or 0)
                        if not nbs_name:
                            continue
                        if nbs_name not in nbs_to_sw or weight > nbs_weights.get(nbs_name, -1):
                            nbs_to_sw[nbs_name] = sw_name
                            nbs_weights[nbs_name] = weight
                nbs_names = sorted(nbs_to_sw.keys(), key=len, reverse=True)
            except Exception:
                pass

        self._nbs_industry_cache = (nbs_to_sw, nbs_names)
        return self._nbs_industry_cache

    def _map_by_nbs_industry(self, name: str) -> Optional[str]:
        if not name:
            return None
        nbs_to_sw, nbs_names = self._load_nbs_industry_mapping()
        if not nbs_to_sw:
            return None

        normalized = self._normalize_nbs_indicator_name(name)
        if not normalized:
            return None

        if normalized in nbs_to_sw:
            return nbs_to_sw[normalized]

        alias = self.nbs_industry_aliases.get(normalized)
        if alias and alias in nbs_to_sw:
            return nbs_to_sw[alias]

        for nbs_name in nbs_names:
            if nbs_name in normalized or normalized in nbs_name:
                return nbs_to_sw[nbs_name]
        return None

    def _load_mapping_config(self):
        if self._mapping_cache is not None:
            return self._mapping_cache

        config_path = PROJECT_ROOT / "config" / "nbs_product_sw_mapping.yaml"
        code_map = {}
        name_map = {}
        keywords = []

        if config_path.exists():
            try:
                import yaml  # type: ignore

                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                for item in cfg.get("products", []) or []:
                    sw = (item or {}).get("sw_industry")
                    if not sw:
                        continue
                    code = (item or {}).get("product_code")
                    name = (item or {}).get("product_name")
                    if code:
                        code_map[str(code).strip()] = sw
                    if name:
                        name_map[self._normalize_product_name(str(name))] = sw

                for item in cfg.get("keywords", []) or []:
                    sw = (item or {}).get("sw_industry")
                    kws = (item or {}).get("keywords") or []
                    kws = [str(k).strip() for k in kws if str(k).strip()]
                    if sw and kws:
                        keywords.append((kws, sw))
            except Exception:
                pass

        self._mapping_cache = (code_map, name_map, keywords)
        return self._mapping_cache

    def parse_nbs_json(self, json_file: str) -> pd.DataFrame:
        """
        解析NBS JSON数据

        Args:
            json_file: JSON文件路径

        Returns:
            DataFrame: 解析后的数据
        """
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "returndata" not in data:
            return pd.DataFrame()

        returndata = data["returndata"]

        # 获取维度节点
        zb_nodes = []  # 指标节点
        sj_nodes = []  # 时间节点

        for node in returndata.get("wdnodes", []):
            if node["wdcode"] == "zb":
                zb_nodes = node["nodes"]
            elif node["wdcode"] == "sj":
                sj_nodes = node["nodes"]

        # 创建指标代码到名称的映射
        zb_code_to_name = {node["code"]: node["name"] for node in zb_nodes}
        {node["code"]: node.get("unit", "") for node in zb_nodes}

        # 创建时间代码到名称的映射
        sj_code_to_name = {node["code"]: node["name"] for node in sj_nodes}

        # 解析数据节点
        records = []
        for datanode in returndata.get("datanodes", []):
            if not datanode["data"]["hasdata"]:
                continue

            # 获取指标代码和时间代码
            wds = datanode["wds"]
            zb_code = None
            sj_code = None

            for wd in wds:
                if wd["wdcode"] == "zb":
                    zb_code = wd["valuecode"]
                elif wd["wdcode"] == "sj":
                    sj_code = wd["valuecode"]

            if zb_code and sj_code:
                value = datanode["data"]["data"]
                records.append(
                    {
                        "product_code": zb_code,
                        "product_name": zb_code_to_name.get(zb_code, ""),
                        "time_code": sj_code,
                        "time_name": sj_code_to_name.get(sj_code, ""),
                        "value": value,
                    }
                )

        df = pd.DataFrame(records)

        # 转换时间代码为日期
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["time_code"].astype(str), format="%Y%m")

        return df

    def assign_sw_industry(self, df: pd.DataFrame, use_nbs_industry: bool = False) -> pd.DataFrame:
        """
        为产品数据添加申万行业映射（不丢弃未匹配项）
        """
        if len(df) == 0:
            return pd.DataFrame()
        df = df.copy()
        code_map, name_map, keyword_overrides = self._load_mapping_config()
        keyword_rules = keyword_overrides + list(self.product_keywords)

        def _map_row(row):
            code = row.get("product_code")
            if code is not None:
                code_str = str(code).strip()
                if code_str in code_map:
                    return code_map[code_str], "config_code"
                if code_str in self.product_to_sw_industry:
                    return self.product_to_sw_industry[code_str], "default_code"

            name = row.get("product_name") or row.get("product")
            name_norm = self._normalize_product_name(name)
            if name_norm in name_map:
                return name_map[name_norm], "config_name"

            for keywords, industry in keyword_rules:
                if any(k in name_norm for k in keywords):
                    return industry, "keyword"

            if use_nbs_industry:
                nbs_sw = self._map_by_nbs_industry(name)
                if nbs_sw:
                    return nbs_sw, "nbs_industry"
            return None, None

        mapped = df.apply(_map_row, axis=1, result_type="expand")
        df["sw_industry"] = mapped[0]
        df["sw_map_source"] = mapped[1]
        return df

    def align_to_sw_industry(self, df: pd.DataFrame, use_nbs_industry: bool = False) -> pd.DataFrame:
        """
        对齐到申万行业（仅保留可映射项）
        """
        if len(df) == 0:
            return pd.DataFrame()
        df = self.assign_sw_industry(df, use_nbs_industry=use_nbs_industry)
        return df[df["sw_industry"].notna()]

    def _infer_industry_by_name(self, name: str) -> Optional[str]:
        if not isinstance(name, str) or not name:
            return None
        for keywords, industry in self.product_keywords:
            if any(k in name for k in keywords):
                return industry
        return None

    def _resolve_macro_dir(self) -> Path:
        if self.data_dir.exists():
            return self.data_dir
        fallback = PROJECT_ROOT / "data" / "tushare" / "macro"
        if fallback.exists():
            return fallback
        fallback_raw = PROJECT_ROOT / "data" / "raw" / "tushare" / "macro"
        if fallback_raw.exists():
            return fallback_raw
        return self.data_dir

    def load_output_csv(self, macro_dir: Path) -> Optional[pd.DataFrame]:
        """
        加载工业品产量CSV（优先nbs_output_2020.csv，其次nbs_output_202512.csv）
        """
        candidates = [
            macro_dir / "nbs_output_2020.csv",
            macro_dir / "nbs_output_202512.csv",
        ]
        for path in candidates:
            if path.exists():
                df = pd.read_csv(path)
                if "product" in df.columns:
                    df = df.rename(columns={"product": "product_name"})
                return df
        return None

    def load_output_json(self, macro_dir: Path) -> Optional[pd.DataFrame]:
        """
        加载工业品产量JSON（支持A020901/A020902等easyquery导出）
        """
        patterns = ["A020901*.json", "A020902*.json", "A0209*.json"]
        files: List[Path] = []
        for pattern in patterns:
            files.extend(sorted(macro_dir.glob(pattern)))
        if not files:
            return None

        dfs = []
        for path in files:
            df = self.parse_nbs_json(str(path))
            if df is not None and len(df) > 0:
                dfs.append(df)
        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

    def calculate_growth_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算增长率

        Args:
            df: 包含原始值的数据

        Returns:
            DataFrame: 包含增长率的数据
        """
        if len(df) == 0:
            return df

        # 按产品和时间排序
        df = df.sort_values(["product_code", "date"]).reset_index(drop=True)

        # 计算同比增长率
        df["yoy"] = df.groupby("product_code")["value"].pct_change(periods=12) * 100

        # 计算环比增长率
        df["mom"] = df.groupby("product_code")["value"].pct_change() * 100

        return df

    def build_mapping_audit(self, df: pd.DataFrame, label: str = "output") -> Dict[str, pd.DataFrame]:
        """
        构建映射审计报告（覆盖率/行业统计/未映射清单）
        """
        if df is None or len(df) == 0:
            return {}

        work = df.copy()
        if "product_name" not in work.columns:
            return {}

        work["product_name_norm"] = work["product_name"].apply(self._normalize_product_name)
        work = work[work["product_name_norm"].notna()]

        total_rows = len(work)
        total_products = work["product_name_norm"].nunique()

        mapped = work[work["sw_industry"].notna()]
        unmapped = work[work["sw_industry"].isna()]

        mapped_rows = len(mapped)
        mapped_products = mapped["product_name_norm"].nunique()
        unmapped_products = unmapped["product_name_norm"].nunique()
        coverage = mapped_products / total_products if total_products else 0.0

        summary = pd.DataFrame(
            [
                {"metric": "total_rows", "value": total_rows},
                {"metric": "unique_products", "value": total_products},
                {"metric": "mapped_rows", "value": mapped_rows},
                {"metric": "mapped_unique_products", "value": mapped_products},
                {"metric": "unmapped_unique_products", "value": unmapped_products},
                {"metric": "coverage_unique_products", "value": round(coverage, 6)},
            ]
        )

        by_industry = (
            mapped.groupby("sw_industry")
            .agg(
                product_count=("product_name_norm", "nunique"),
                row_count=("sw_industry", "size"),
            )
            .reset_index()
            .sort_values(["product_count", "row_count"], ascending=False)
        )

        if "date" in work.columns:
            unmatched = (
                unmapped.groupby(["product_name_norm", "product_code"], dropna=False)
                .agg(
                    row_count=("product_name_norm", "size"),
                    start_date=("date", "min"),
                    end_date=("date", "max"),
                )
                .reset_index()
                .rename(columns={"product_name_norm": "product_name"})
            )
        else:
            unmatched = (
                unmapped.groupby(["product_name_norm", "product_code"], dropna=False)
                .size()
                .reset_index(name="row_count")
                .rename(columns={"product_name_norm": "product_name"})
            )

        unmatched["has_chinese"] = unmatched["product_name"].apply(
            lambda x: bool(re.search(r"[\u4e00-\u9fff]", str(x)))
        )

        return {
            "summary": summary,
            "by_industry": by_industry,
            "unmatched": unmatched.sort_values(["has_chinese", "row_count"], ascending=False),
            "label": label,
        }

    def aggregate_to_industry_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        聚合到行业级别

        Args:
            df: 对齐后的数据

        Returns:
            DataFrame: 行业级别数据
        """
        if len(df) == 0:
            return pd.DataFrame()

        agg_spec = {}
        if "yoy" in df.columns:
            agg_spec["yoy"] = "mean"
        if "mom" in df.columns:
            agg_spec["mom"] = "mean"
        if "value" in df.columns:
            agg_spec["value"] = "sum"
        if not agg_spec:
            return pd.DataFrame()

        # 按行业和日期分组，计算平均值
        industry_df = df.groupby(["sw_industry", "date"]).agg(agg_spec).reset_index()

        # 重命名列（仅存在时）
        rename_map = {}
        if "yoy" in industry_df.columns:
            rename_map["yoy"] = "output_yoy"
        if "mom" in industry_df.columns:
            rename_map["mom"] = "output_mom"
        if rename_map:
            industry_df = industry_df.rename(columns=rename_map)

        return industry_df

    def process_all(self) -> Dict[str, pd.DataFrame]:
        """
        处理所有NBS数据

        Returns:
            Dict: 包含所有处理后的数据
        """
        print("=" * 80)
        print("NBS工业品数据对齐")
        print("=" * 80)

        results = {}
        macro_dir = self._resolve_macro_dir()

        # 1. 处理工业品产量数据
        print("\n1. 处理工业品产量数据...")
        output_df = self.load_output_csv(macro_dir)
        if output_df is None or len(output_df) == 0:
            output_df = self.load_output_json(macro_dir)
        if output_df is None:
            output_file = macro_dir / "A020901_工业品产量.json"
            if output_file.exists():
                output_df = self.parse_nbs_json(str(output_file))
        if output_df is not None and len(output_df) > 0:
            print(f"  原始数据: {len(output_df)}条记录")

            if "date" in output_df.columns:
                output_df["date"] = pd.to_datetime(output_df["date"])
            else:
                output_df["date"] = pd.to_datetime(output_df["time_code"].astype(str), format="%Y%m")

            # 计算增长率
            if "output_value" in output_df.columns:
                output_df = output_df.rename(columns={"output_value": "value"})
            output_df = self.calculate_growth_rate(output_df)

            # 产品级（保留全部产品）
            product_level_df = self.assign_sw_industry(output_df, use_nbs_industry=False)

            # 对齐到申万行业（行业代理）
            output_df = self.align_to_sw_industry(output_df, use_nbs_industry=False)
            print(f"  对齐后: {len(output_df)}条记录")

            # 聚合到行业级别
            industry_output_df = self.aggregate_to_industry_level(output_df)
            print(f"  行业级别: {len(industry_output_df)}条记录")

            results["output"] = {"raw": output_df, "product": product_level_df, "industry": industry_output_df}

        # 2. 处理固定资产投资数据
        print("\n2. 处理固定资产投资数据...")
        fai_file = macro_dir / "A0403_固定资产投资.json"
        if fai_file.exists():
            fai_df = self.parse_nbs_json(str(fai_file))
            print(f"  原始数据: {len(fai_df)}条记录")

            # 计算增长率
            fai_df = self.calculate_growth_rate(fai_df)

            # 对齐到申万行业
            fai_df = self.align_to_sw_industry(fai_df, use_nbs_industry=True)
            print(f"  对齐后: {len(fai_df)}条记录")

            # 聚合到行业级别
            industry_fai_df = self.aggregate_to_industry_level(fai_df)
            print(f"  行业级别: {len(industry_fai_df)}条记录")

            results["fai"] = {"raw": fai_df, "industry": industry_fai_df}

        # 3. 处理价格指数数据
        print("\n3. 处理价格指数数据...")
        price_file = macro_dir / "A010D02_价格指数.json"
        if price_file.exists():
            price_df = self.parse_nbs_json(str(price_file))
            print(f"  原始数据: {len(price_df)}条记录")

            # 价格指数本身就是增长率形式，不需要额外计算
            price_df["yoy"] = price_df["value"] - 100  # 转换为百分比形式

            # 对齐到申万行业
            price_df = self.align_to_sw_industry(price_df, use_nbs_industry=True)
            print(f"  对齐后: {len(price_df)}条记录")

            # 聚合到行业级别
            industry_price_df = self.aggregate_to_industry_level(price_df)
            print(f"  行业级别: {len(industry_price_df)}条记录")

            results["price"] = {"raw": price_df, "industry": industry_price_df}

        # 4. 合并所有数据
        print("\n4. 合并所有数据...")
        all_industry_data = []

        for data_type, data_dict in results.items():
            if "industry" in data_dict and len(data_dict["industry"]) > 0:
                df = data_dict["industry"].copy()
                df["data_type"] = data_type

                # 统一列名
                if "output_yoy" in df.columns:
                    df["yoy"] = df["output_yoy"]
                if "fai_yoy" in df.columns:
                    df["yoy"] = df["fai_yoy"]
                if "yoy" in df.columns and "output_yoy" not in df.columns and data_type == "output":
                    df["output_yoy"] = df["yoy"]

                if "output_mom" in df.columns:
                    df["mom"] = df["output_mom"]
                if "fai_mom" in df.columns:
                    df["mom"] = df["fai_mom"]
                if "mom" in df.columns and "output_mom" not in df.columns and data_type == "output":
                    df["output_mom"] = df["mom"]

                all_industry_data.append(df)

        if all_industry_data:
            combined_df = pd.concat(all_industry_data, ignore_index=True)
            print(f"  合并后: {len(combined_df)}条记录")
            print(f"  可用列: {combined_df.columns.tolist()}")

            # 检查可用的数值列
            value_cols = [col for col in combined_df.columns if col in ["yoy", "mom", "value"]]
            print(f"  数值列: {value_cols}")

            if value_cols:
                # 透视表格式
                pivot_df = combined_df.pivot_table(
                    index=["sw_industry", "date"], columns="data_type", values=value_cols, aggfunc="mean"
                )

                # 展平列名
                pivot_df.columns = [f"{data_type}_{metric}" for metric, data_type in pivot_df.columns]
                pivot_df = pivot_df.reset_index()

                print(f"  透视表: {len(pivot_df)}条记录")
                print(f"  列名: {pivot_df.columns.tolist()}")

                results["combined"] = pivot_df
            else:
                print("  警告: 没有可用的数值列")
                results["combined"] = pd.DataFrame()

        print("\n" + "=" * 80)
        print("数据对齐完成")
        print("=" * 80)

        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="宏观数据目录")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    aligner = NBSIndustrialDataAligner(data_dir=args.data_dir)
    results = aligner.process_all()

    # 保存结果
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if "combined" in results:
        output_file = os.path.join(output_dir, "nbs_industrial_aligned.parquet")
        results["combined"].to_parquet(output_file, index=False)
        print(f"\n对齐后的数据已保存到: {output_file}")

        # 显示样例数据
        print("\n样例数据:")
        print(results["combined"].head(10))

    def save_audit(audit_result: Dict[str, pd.DataFrame]):
        label = audit_result.get("label", "output")

        summary_df = audit_result.get("summary")
        if summary_df is not None and len(summary_df) > 0:
            summary_file = os.path.join(output_dir, f"nbs_{label}_mapping_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"映射覆盖率摘要已保存: {summary_file}")

        by_industry_df = audit_result.get("by_industry")
        if by_industry_df is not None and len(by_industry_df) > 0:
            by_industry_file = os.path.join(output_dir, f"nbs_{label}_mapping_by_industry.csv")
            by_industry_df.to_csv(by_industry_file, index=False)
            print(f"行业映射统计已保存: {by_industry_file}")

        unmatched_df = audit_result.get("unmatched")
        if unmatched_df is not None and len(unmatched_df) > 0:
            unmatched_file = os.path.join(output_dir, f"nbs_{label}_mapping_unmatched.csv")
            unmatched_df.to_csv(unmatched_file, index=False)
            print(f"未映射清单已保存: {unmatched_file}")

    if "output" in results and isinstance(results["output"], dict):
        product_df = results["output"].get("product")
        if product_df is not None and len(product_df) > 0:
            product_file = os.path.join(output_dir, "nbs_output_product_level.parquet")
            product_df.to_parquet(product_file, index=False)
            print(f"\n产品级产量数据已保存到: {product_file}")

            audit = aligner.build_mapping_audit(product_df, label="output")
            save_audit(audit)

    if "fai" in results and isinstance(results["fai"], dict):
        fai_df = results["fai"].get("raw")
        if fai_df is not None and len(fai_df) > 0:
            audit = aligner.build_mapping_audit(fai_df, label="fai")
            save_audit(audit)

    if "price" in results and isinstance(results["price"], dict):
        price_df = results["price"].get("raw")
        if price_df is not None and len(price_df) > 0:
            audit = aligner.build_mapping_audit(price_df, label="price")
            save_audit(audit)


if __name__ == "__main__":
    main()
