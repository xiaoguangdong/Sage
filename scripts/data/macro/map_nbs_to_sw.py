#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将NBS宏观数据映射到申万行业

功能：
1. 读取NBS PPI数据
2. 使用映射配置将NBS数据映射到申万行业
3. 保存映射后的申万行业数据

作者：iFlow CLI
日期：2026-02-11
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.macro.industry_mapper import IndustryMapper

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/macro/map_nbs_to_sw.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class NBSMapper:
    """NBS数据映射器"""

    def __init__(self):
        """初始化映射器"""
        self.mapper = IndustryMapper()
        self.data_dir = project_root / "data" / "tushare" / "macro"
        self.output_dir = project_root / "data" / "tushare" / "macro"

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_nbs_ppi_data(self, file_path: str = None) -> pd.DataFrame:
        """
        加载NBS PPI数据

        Args:
            file_path: NBS PPI数据文件路径

        Returns:
            NBS PPI数据DataFrame
        """
        if file_path is None:
            # 查找最新的NBS PPI数据文件
            nbs_files = list(self.data_dir.glob("nbs_ppi_industry_*.csv"))
            if not nbs_files:
                raise FileNotFoundError("未找到NBS PPI数据文件")

            # 按文件名排序，取最新的
            file_path = max(nbs_files, key=lambda f: f.stat().st_mtime)

        logger.info(f"加载NBS PPI数据: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"  ✓ 加载成功: {len(df)} 条记录")

        return df

    def map_nbs_to_sw(self, nbs_data: pd.DataFrame, value_col: str = "ppi_yoy") -> pd.DataFrame:
        """
        将NBS数据映射到申万行业

        Args:
            nbs_data: NBS数据DataFrame
            value_col: 数值列名

        Returns:
            映射后的申万行业DataFrame
        """
        logger.info("开始映射NBS数据到申万行业...")

        # 使用映射器进行映射
        sw_data = self.mapper.map_nbs_data_to_sw(
            nbs_data, value_col=value_col, date_col="date", nbs_name_col="industry"
        )

        logger.info(f"  ✓ 映射完成: {len(sw_data)} 个申万行业")

        return sw_data

    def save_sw_data(self, sw_data: pd.DataFrame, value_col: str = "ppi_yoy") -> str:
        """
        保存申万行业数据

        Args:
            sw_data: 申万行业数据DataFrame
            value_col: 数值列名

        Returns:
            保存的文件路径
        """
        # 生成文件名（包含日期）
        if len(sw_data) > 0:
            date_str = sw_data["date"].iloc[0]
            date_str = date_str.replace("-", "")[:6]  # 只保留年月
        else:
            date_str = datetime.now().strftime("%Y%m")

        filename = f"sw_{value_col}_{date_str}.csv"
        file_path = self.output_dir / filename

        # 保存为CSV
        sw_data.to_csv(file_path, index=False, encoding="utf-8-sig")
        logger.info(f"  ✓ 数据已保存: {file_path}")

        return str(file_path)

    def process(self, nbs_file: str = None, value_col: str = "ppi_yoy") -> pd.DataFrame:
        """
        处理流程：加载 -> 映射 -> 保存

        Args:
            nbs_file: NBS数据文件路径
            value_col: 数值列名

        Returns:
            映射后的申万行业DataFrame
        """
        logger.info("=" * 60)
        logger.info("开始处理NBS到申万行业的映射")
        logger.info("=" * 60)

        # 1. 加载NBS数据
        nbs_data = self.load_nbs_ppi_data(nbs_file)

        # 2. 映射到申万行业
        sw_data = self.map_nbs_to_sw(nbs_data, value_col)

        # 3. 保存申万数据
        if len(sw_data) > 0:
            self.save_sw_data(sw_data, value_col)

        logger.info("=" * 60)
        logger.info("映射处理完成")
        logger.info("=" * 60)

        return sw_data

    def display_results(self, sw_data: pd.DataFrame):
        """显示映射结果"""
        print("\n" + "=" * 60)
        print("申万行业映射结果")
        print("=" * 60)
        print(sw_data.to_string(index=False))

        # 统计信息
        print("\n" + "=" * 60)
        print("统计信息")
        print("=" * 60)
        print(f"申万行业数量: {len(sw_data)}")

        if "sw_ppi_yoy" in sw_data.columns:
            valid_data = sw_data[sw_data["sw_ppi_yoy"].notna()]
            print(f"有效数据数量: {len(valid_data)}")
            if len(valid_data) > 0:
                print(f"平均值: {valid_data['sw_ppi_yoy'].mean():.2f}")
                print(f"最小值: {valid_data['sw_ppi_yoy'].min():.2f}")
                print(f"最大值: {valid_data['sw_ppi_yoy'].max():.2f}")
                print(f"标准差: {valid_data['sw_ppi_yoy'].std():.2f}")


def main():
    """主函数"""
    # 创建映射器
    mapper = NBSMapper()

    # 处理NBS PPI数据
    sw_data = mapper.process()

    # 显示结果
    mapper.display_results(sw_data)


if __name__ == "__main__":
    main()
