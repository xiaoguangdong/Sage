#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Northbound Hold 数据质量验证脚本

验证内容：
1. 时间覆盖分析（找出缺口）
2. 数据量统计（每年的记录数）
3. 字段完整性
4. 异常值检测
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def validate_northbound_data(data_root: Path) -> Dict:
    """验证 northbound_hold 数据质量

    Args:
        data_root: Tushare 数据根目录

    Returns:
        验证结果字典
    """
    northbound_path = data_root / "northbound" / "northbound_hold.parquet"

    if not northbound_path.exists():
        return {"status": "error", "message": f"Northbound 数据文件不存在: {northbound_path}"}

    try:
        df = pd.read_parquet(northbound_path)

        # 转换日期
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

        results = {
            "status": "success",
            "total_records": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": df["trade_date"].min().strftime("%Y-%m-%d"),
                "end": df["trade_date"].max().strftime("%Y-%m-%d"),
            },
            "yearly_stats": {},
            "gaps": [],
            "data_quality": {},
        }

        # 按年份统计
        df["year"] = df["trade_date"].dt.year
        yearly_counts = df.groupby("year").size().to_dict()
        results["yearly_stats"] = yearly_counts

        # 检测时间缺口
        all_dates = pd.date_range(start=df["trade_date"].min(), end=df["trade_date"].max(), freq="D")
        existing_dates = set(df["trade_date"].dt.date)

        # 找出连续缺失的时间段
        gaps = []
        gap_start = None
        for date in all_dates:
            if date.date() not in existing_dates:
                if gap_start is None:
                    gap_start = date
            else:
                if gap_start is not None:
                    gap_days = (date - gap_start).days
                    if gap_days > 30:  # 只记录超过30天的缺口
                        gaps.append(
                            {
                                "start": gap_start.strftime("%Y-%m-%d"),
                                "end": (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                                "days": gap_days,
                            }
                        )
                    gap_start = None

        results["gaps"] = gaps

        # 数据质量检查
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        results["data_quality"] = {
            "has_inf": df[numeric_cols].isin([np.inf, -np.inf]).any().any(),
            "has_nan": df[numeric_cols].isna().any().any(),
            "nan_counts": df[numeric_cols].isna().sum().to_dict(),
            "unique_stocks": df["ts_code"].nunique() if "ts_code" in df.columns else 0,
        }

        return results

    except Exception as e:
        return {"status": "error", "message": f"读取数据失败: {str(e)}"}


def print_validation_report(results: Dict):
    """打印验证报告"""
    print("\n" + "=" * 60)
    print("Northbound Hold 数据质量验证报告")
    print("=" * 60)

    if results.get("status") == "error":
        print(f"❌ 错误: {results['message']}")
        return

    print("\n基本信息:")
    print(f"  总记录数: {results['total_records']:,}")
    print(f"  数据范围: {results['date_range']['start']} ~ {results['date_range']['end']}")
    print(f"  唯一股票数: {results['data_quality']['unique_stocks']}")

    print("\n按年份统计:")
    for year in sorted(results["yearly_stats"].keys()):
        count = results["yearly_stats"][year]
        print(f"  {year}: {count:,} 条记录")

    # 检查缺口
    if results["gaps"]:
        print(f"\n⚠️  发现 {len(results['gaps'])} 个时间缺口（>30天）:")
        for gap in results["gaps"]:
            print(f"  - {gap['start']} ~ {gap['end']} ({gap['days']} 天)")
    else:
        print("\n✅ 未发现明显时间缺口")

    # 数据质量
    quality = results["data_quality"]
    print("\n数据质量:")
    print(f"  包含无穷值: {'是' if quality['has_inf'] else '否'}")
    print(f"  包含缺失值: {'是' if quality['has_nan'] else '否'}")

    if quality["has_nan"]:
        print("  缺失值统计:")
        for col, count in quality["nan_counts"].items():
            if count > 0:
                print(f"    - {col}: {count} 个缺失值")

    # 关键发现
    print("\n关键发现:")
    years = sorted(results["yearly_stats"].keys())
    if len(years) > 1:
        year_gaps = []
        for i in range(len(years) - 1):
            if years[i + 1] - years[i] > 1:
                year_gaps.append((years[i], years[i + 1]))

        if year_gaps:
            print("  ⚠️  发现年份缺口:")
            for start_year, end_year in year_gaps:
                missing_years = list(range(start_year + 1, end_year))
                print(f"    - {start_year} 和 {end_year} 之间缺失: {missing_years}")
        else:
            print("  ✅ 年份连续，无明显缺口")

    print("\n" + "=" * 60)


def main():
    data_root = Path("data/tushare")

    print("开始验证 northbound_hold 数据...")
    results = validate_northbound_data(data_root)

    # 打印报告
    print_validation_report(results)

    # 保存结果
    output_path = Path("logs/data/northbound_validation_report.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Northbound Hold 数据验证报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总记录数: {results.get('total_records', 0):,}\n")
        f.write(f"数据范围: {results.get('date_range', {}).get('start', 'N/A')} ~ ")
        f.write(f"{results.get('date_range', {}).get('end', 'N/A')}\n\n")

        if results.get("yearly_stats"):
            f.write("按年份统计:\n")
            for year in sorted(results["yearly_stats"].keys()):
                f.write(f"  {year}: {results['yearly_stats'][year]:,}\n")

        if results.get("gaps"):
            f.write(f"\n时间缺口 ({len(results['gaps'])} 个):\n")
            for gap in results["gaps"]:
                f.write(f"  {gap['start']} ~ {gap['end']} ({gap['days']} 天)\n")

    print(f"\n验证报告已保存到: {output_path}")


if __name__ == "__main__":
    main()
