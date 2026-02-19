#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moneyflow 数据质量验证脚本

验证内容：
1. 文件完整性（所有股票文件是否存在）
2. 时间覆盖（每个股票的起止日期）
3. 数据质量（异常值、缺失值）
4. 字段完整性
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def validate_moneyflow_data(data_root: Path) -> Dict:
    """验证 moneyflow 数据质量

    Args:
        data_root: Tushare 数据根目录

    Returns:
        验证结果字典
    """
    moneyflow_dir = data_root / "moneyflow"

    if not moneyflow_dir.exists():
        return {"status": "error", "message": f"Moneyflow 目录不存在: {moneyflow_dir}"}

    # 获取所有 parquet 文件
    files = list(moneyflow_dir.glob("*.parquet"))

    print(f"找到 {len(files)} 个 moneyflow 数据文件")

    results = {"total_files": len(files), "valid_files": 0, "invalid_files": 0, "file_details": [], "summary": {}}

    all_start_dates = []
    all_end_dates = []
    total_records = 0

    for file_path in files:
        ts_code = file_path.stem

        try:
            df = pd.read_parquet(file_path)

            # 基本统计
            n_records = len(df)
            total_records += n_records

            # 时间范围
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                start_date = df["trade_date"].min()
                end_date = df["trade_date"].max()
                all_start_dates.append(start_date)
                all_end_dates.append(end_date)
            else:
                start_date = None
                end_date = None

            # 检查必需字段
            required_fields = [
                "trade_date",
                "buy_sm_amount",
                "sell_sm_amount",
                "buy_md_amount",
                "sell_md_amount",
                "buy_lg_amount",
                "sell_lg_amount",
                "buy_elg_amount",
                "sell_elg_amount",
            ]
            missing_fields = [f for f in required_fields if f not in df.columns]

            # 检查异常值
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            has_inf = df[numeric_cols].isin([np.inf, -np.inf]).any().any()
            has_nan = df[numeric_cols].isna().any().any()

            file_info = {
                "ts_code": ts_code,
                "records": n_records,
                "start_date": start_date.strftime("%Y-%m-%d") if start_date else None,
                "end_date": end_date.strftime("%Y-%m-%d") if end_date else None,
                "missing_fields": missing_fields,
                "has_inf": has_inf,
                "has_nan": has_nan,
                "status": "valid" if not missing_fields and not has_inf else "warning",
            }

            results["file_details"].append(file_info)

            if file_info["status"] == "valid":
                results["valid_files"] += 1
            else:
                results["invalid_files"] += 1

        except Exception as e:
            results["file_details"].append({"ts_code": ts_code, "status": "error", "error": str(e)})
            results["invalid_files"] += 1

    # 汇总统计
    if all_start_dates:
        results["summary"] = {
            "total_records": total_records,
            "earliest_date": min(all_start_dates).strftime("%Y-%m-%d"),
            "latest_date": max(all_end_dates).strftime("%Y-%m-%d"),
            "avg_records_per_stock": total_records / len(files) if files else 0,
        }

    return results


def print_validation_report(results: Dict):
    """打印验证报告"""
    print("\n" + "=" * 60)
    print("Moneyflow 数据质量验证报告")
    print("=" * 60)

    if results.get("status") == "error":
        print(f"❌ 错误: {results['message']}")
        return

    print("\n文件统计:")
    print(f"  总文件数: {results['total_files']}")
    print(f"  有效文件: {results['valid_files']}")
    print(f"  异常文件: {results['invalid_files']}")

    if results.get("summary"):
        summary = results["summary"]
        print("\n数据覆盖:")
        print(f"  总记录数: {summary['total_records']:,}")
        print(f"  最早日期: {summary['earliest_date']}")
        print(f"  最晚日期: {summary['latest_date']}")
        print(f"  平均记录数/股票: {summary['avg_records_per_stock']:.0f}")

    # 显示异常文件
    invalid_files = [f for f in results["file_details"] if f["status"] != "valid"]
    if invalid_files:
        print(f"\n⚠️  发现 {len(invalid_files)} 个异常文件:")
        for f in invalid_files[:10]:  # 只显示前10个
            print(f"  - {f['ts_code']}: {f.get('error', '字段缺失或异常值')}")
        if len(invalid_files) > 10:
            print(f"  ... 还有 {len(invalid_files) - 10} 个异常文件")
    else:
        print("\n✅ 所有文件验证通过")

    print("\n" + "=" * 60)


def save_detailed_report(results: Dict, output_path: Path):
    """保存详细报告到 CSV"""
    if results.get("file_details"):
        df = pd.DataFrame(results["file_details"])
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n详细报告已保存到: {output_path}")


def main():
    data_root = Path("data/tushare")

    print("开始验证 moneyflow 数据...")
    results = validate_moneyflow_data(data_root)

    # 打印报告
    print_validation_report(results)

    # 保存详细报告
    output_path = Path("logs/data/moneyflow_validation_report.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_detailed_report(results, output_path)


if __name__ == "__main__":
    main()
