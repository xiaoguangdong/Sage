#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
申万-NBS行业映射工具

功能：
1. 加载申万-NBS行业映射配置
2. 将NBS宏观数据映射到申万行业
3. 计算申万行业的合成指标

作者：iFlow CLI
日期：2026-02-11
"""

from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from scripts.data._shared.runtime import setup_logger

logger = setup_logger("industry_mapper", module="macro")


class IndustryMapper:
    """申万-NBS行业映射器"""

    def __init__(self, config_path: Optional[str] = None, level: str = "L1"):
        """
        初始化映射器

        Args:
            config_path: 映射配置文件路径，默认根据level自动选择
            level: 行业级别，"L1"表示一级行业，"L2"表示二级行业
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent
            if level == "L2":
                config_path = project_root / "config" / "sw_nbs_mapping_l2.yaml"
            else:
                config_path = project_root / "config" / "sw_nbs_mapping.yaml"

        self.config_path = config_path
        self.level = level
        self.config = self._load_config()

        # 根据配置文件加载映射
        if level == "L2":
            self.sw_to_nbs_map = self.config.get("sw_l2_to_nbs", {})
        else:
            self.sw_to_nbs_map = self.config.get("sw_to_nbs", {})

        logger.info(f"加载映射配置: {config_path}")
        logger.info(f"申万{self.level}级行业数量: {len(self.sw_to_nbs_map)}")

    def _load_config(self) -> Dict:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"配置文件不存在: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"配置文件解析错误: {e}")
            raise

    def get_sw_industries(self) -> List[str]:
        """获取所有申万一级行业名称"""
        return list(self.sw_to_nbs_map.keys())

    def get_nbs_mapping(self, sw_industry: str) -> Optional[List[Dict]]:
        """
        获取申万行业对应的NBS行业映射

        Args:
            sw_industry: 申万行业名称

        Returns:
            NBS行业映射列表，包含nbs_industry, weight, nbs_code
        """
        return self.sw_to_nbs_map.get(sw_industry)

    def fuzzy_match_nbs(
        self, nbs_name: str, nbs_data: pd.DataFrame, nbs_name_col: str = "industry", threshold: float = 0.6
    ) -> Optional[str]:
        """
        模糊匹配NBS行业名称

        Args:
            nbs_name: 待匹配的NBS行业名称
            nbs_data: NBS数据DataFrame
            nbs_name_col: NBS行业名称列名
            threshold: 匹配阈值，默认0.6

        Returns:
            匹配到的NBS行业名称，None表示未匹配
        """
        best_match = None
        best_ratio = 0

        for existing_name in nbs_data[nbs_name_col].unique():
            ratio = SequenceMatcher(None, nbs_name, existing_name).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = existing_name

        return best_match

    def map_nbs_data_to_sw(
        self, nbs_data: pd.DataFrame, value_col: str = "ppi_yoy", date_col: str = "date", nbs_name_col: str = "industry"
    ) -> pd.DataFrame:
        """
        将NBS宏观数据映射到申万行业

        Args:
            nbs_data: NBS宏观数据DataFrame，必须包含industry和value_col列
            value_col: 数值列名，如'ppi_yoy'
            date_col: 日期列名
            nbs_name_col: NBS行业名称列名

        Returns:
            映射后的申万行业DataFrame，包含sw_industry, value, date等列
        """
        results = []

        for sw_industry in self.sw_to_nbs_map:
            nbs_mappings = self.sw_to_nbs_map[sw_industry]

            # 计算该申万行业的合成指标
            sw_values = []
            sw_dates = []
            sw_weights = []

            for nbs_mapping in nbs_mappings:
                nbs_name = nbs_mapping["nbs_industry"]
                weight = nbs_mapping["weight"]

                # 从NBS数据中查找对应的行业
                nbs_row = nbs_data[nbs_data[nbs_name_col] == nbs_name]

                if len(nbs_row) == 0:
                    # 尝试模糊匹配
                    matched_name = self.fuzzy_match_nbs(nbs_name, nbs_data, nbs_name_col)
                    if matched_name:
                        nbs_row = nbs_data[nbs_data[nbs_name_col] == matched_name]

                if len(nbs_row) > 0:
                    nbs_value = nbs_row[value_col].values[0]
                    nbs_date = nbs_row[date_col].values[0]

                    # 检查数值是否有效
                    if pd.notna(nbs_value) and not np.isinf(nbs_value):
                        sw_values.append(nbs_value * weight)
                        sw_dates.append(nbs_date)
                        sw_weights.append(weight)

            if sw_values:
                # 加权平均
                total_weight = sum(sw_weights)
                sw_value = sum(sw_values) / total_weight if total_weight > 0 else np.nan

                # 获取日期（假设所有NBS数据日期相同）
                sw_date = sw_dates[0] if sw_dates else None

                results.append(
                    {
                        "sw_industry": sw_industry,
                        f"sw_{value_col}": sw_value,
                        "date": sw_date,
                        "source_nbs_count": len(sw_values),
                    }
                )

        return pd.DataFrame(results)

    def get_nbs_coverage(self, sw_industry: str) -> float:
        """
        获取申万行业的NBS覆盖率

        Args:
            sw_industry: 申万行业名称

        Returns:
            NBS覆盖率（0-1之间）
        """
        nbs_mappings = self.sw_to_nbs_map.get(sw_industry, [])
        if not nbs_mappings:
            return 0.0

        # 这里简化为权重之和
        total_weight = sum(m["weight"] for m in nbs_mappings)
        return min(total_weight, 1.0)

    def validate_mapping(self) -> Dict:
        """
        验证映射配置的完整性

        Returns:
            验证结果字典
        """
        results = {"valid": True, "errors": [], "warnings": []}

        # 检查每个申万行业的映射
        for sw_industry, nbs_mappings in self.sw_to_nbs_map.items():
            # 检查权重总和
            total_weight = sum(m["weight"] for m in nbs_mappings)
            if abs(total_weight - 1.0) > 0.01:  # 允许1%的误差
                results["warnings"].append(f"{sw_industry}: 权重总和为{total_weight:.2f}，不等于1.0")

            # 检查NBS行业名称是否为空
            for nbs_mapping in nbs_mappings:
                nbs_name = nbs_mapping["nbs_industry"]
                if not nbs_name:
                    results["errors"].append(f"{sw_industry}: NBS行业名称为空")
        results["valid"] = len(results["errors"]) == 0
        return results

    def get_mapping_summary(self) -> pd.DataFrame:
        """
        获取映射摘要

        Returns:
            映射摘要DataFrame
        """
        summary = []

        for sw_industry, nbs_mappings in self.sw_to_nbs_map.items():
            total_weight = sum(m["weight"] for m in nbs_mappings)
            nbs_industries = [m["nbs_industry"] for m in nbs_mappings]

            summary.append(
                {
                    "sw_industry": sw_industry,
                    "nbs_count": len(nbs_mappings),
                    "total_weight": total_weight,
                    "nbs_industries": ", ".join(nbs_industries),
                }
            )

        return pd.DataFrame(summary)


def main():
    """测试函数"""
    # 初始化映射器
    mapper = IndustryMapper()

    # 打印申万行业列表
    print("\n=== 申万行业列表 ===")
    sw_industries = mapper.get_sw_industries()
    for i, industry in enumerate(sw_industries, 1):
        print(f"{i}. {industry}")

    # 打印映射摘要
    print("\n=== 映射摘要 ===")
    summary = mapper.get_mapping_summary()
    print(summary.to_string(index=False))

    # 验证映射
    print("\n=== 映射验证 ===")
    validation = mapper.validate_mapping()
    if validation["valid"]:
        print("✓ 映射配置有效")
    else:
        print("✗ 映射配置有错误:")
        for error in validation["errors"]:
            print(f"  - {error}")

    if validation["warnings"]:
        print("⚠ 映射配置有警告:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    # 测试映射功能
    print("\n=== 测试NBS数据映射 ===")

    # 创建模拟NBS数据
    mock_nbs_data = pd.DataFrame(
        [
            {"industry": "煤炭开采和洗选业", "industry_code": "C01", "ppi_yoy": -2.5, "date": "2025-12-01"},
            {"industry": "化学原料和化学制品制造业", "industry_code": "C13", "ppi_yoy": -4.2, "date": "2025-12-01"},
            {"industry": "黑色金属冶炼和压延加工业", "industry_code": "C17", "ppi_yoy": -9.5, "date": "2025-12-01"},
            {"industry": "有色金属冶炼和压延加工业", "industry_code": "C18", "ppi_yoy": -4.8, "date": "2025-12-01"},
            {"industry": "汽车制造业", "industry_code": "C22", "ppi_yoy": -1.5, "date": "2025-12-01"},
        ]
    )

    print("\n输入NBS数据:")
    print(mock_nbs_data[["industry", "ppi_yoy"]])

    # 映射到申万行业
    sw_data = mapper.map_nbs_data_to_sw(mock_nbs_data, value_col="ppi_yoy")

    print("\n输出申万行业数据:")
    print(sw_data[["sw_industry", "sw_ppi_yoy", "date", "source_nbs_count"]])


if __name__ == "__main__":
    main()
