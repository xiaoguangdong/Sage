"""
数据缺失值检查脚本
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

# 添加项目路径
sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.data._shared.runtime import log_task_summary, setup_logger

logger = setup_logger("check_missing_values", module="monitoring")


def check_single_file(file_path: str) -> Dict:
    """
    检查单个文件的缺失值

    Args:
        file_path: 文件路径

    Returns:
        缺失值信息字典
    """
    try:
        df = pd.read_parquet(file_path)

        # 基本统计
        total_rows = len(df)
        total_cols = len(df.columns)

        # 缺失值统计
        missing_info = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / total_rows * 100 if total_rows > 0 else 0

            if missing_count > 0:
                missing_info[col] = {"count": int(missing_count), "percentage": float(missing_pct)}

        return {
            "file": file_path,
            "total_rows": total_rows,
            "total_cols": total_cols,
            "missing_columns": len(missing_info),
            "missing_details": missing_info,
        }

    except Exception as e:
        logger.error(f"检查文件 {file_path} 时出错: {e}")
        return {"file": file_path, "error": str(e)}


def check_all_files(data_dir: str = "data/baostock", sample_size: int = 100) -> Dict:
    """
    检查所有文件的缺失值

    Args:
        data_dir: 数据目录
        sample_size: 采样检查的文件数量（如果文件太多）

    Returns:
        缺失值统计信息
    """
    logger.info("开始检查数据缺失值...")

    data_path = Path(data_dir)
    parquet_files = list(data_path.glob("*.parquet"))

    logger.info(f"发现 {len(parquet_files)} 个parquet文件")

    # 如果文件太多，进行采样
    if len(parquet_files) > sample_size:
        logger.info(f"文件数量较多，采样检查 {sample_size} 个文件")
        import random

        random.seed(42)
        parquet_files = random.sample(parquet_files, sample_size)

    # 检查每个文件
    results = []
    missing_summary = {}
    files_with_missing = 0

    for i, file_path in enumerate(parquet_files):
        if (i + 1) % 10 == 0:
            logger.info(f"已检查 {i + 1}/{len(parquet_files)} 个文件...")

        result = check_single_file(str(file_path))
        results.append(result)

        if "missing_columns" in result and result["missing_columns"] > 0:
            files_with_missing += 1

            # 统计每列的缺失情况
            for col, info in result["missing_details"].items():
                if col not in missing_summary:
                    missing_summary[col] = {"files_count": 0, "total_missing": 0, "avg_percentage": 0}
                missing_summary[col]["files_count"] += 1
                missing_summary[col]["total_missing"] += info["count"]
                missing_summary[col]["avg_percentage"] += info["percentage"]

    # 计算平均缺失率
    for col in missing_summary:
        if missing_summary[col]["files_count"] > 0:
            missing_summary[col]["avg_percentage"] /= missing_summary[col]["files_count"]

    # 统计总体情况
    total_files = len(results)
    total_rows = sum(r.get("total_rows", 0) for r in results if "total_rows" in r)

    logger.info("缺失值检查完成")
    logger.info(f"检查文件总数: {total_files}")
    logger.info(f"有缺失值的文件数: {files_with_missing} ({files_with_missing/total_files*100:.2f}%)")
    logger.info(f"总数据行数: {total_rows}")

    if missing_summary:
        logger.info("\n缺失值统计（按列）:")
        for col, info in sorted(missing_summary.items(), key=lambda x: x[1]["files_count"], reverse=True):
            logger.info(f"  {col}: {info['files_count']}个文件, 平均缺失率 {info['avg_percentage']:.2f}%")

    return {
        "total_files": total_files,
        "files_with_missing": files_with_missing,
        "missing_percentage": files_with_missing / total_files * 100 if total_files > 0 else 0,
        "total_rows": total_rows,
        "missing_summary": missing_summary,
        "sample_results": results[:10],  # 保存前10个结果作为样本
    }


def check_data_quality_summary(data_dir: str = "data/baostock"):
    """
    生成数据质量摘要报告

    Args:
        data_dir: 数据目录
    """
    logger.info("=" * 60)
    logger.info("数据质量摘要报告")
    logger.info("=" * 60)

    # 检查缺失值
    missing_report = check_all_files(data_dir)

    # 保存报告
    report_path = Path("data/processed/data_quality_report.json")
    import json

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(missing_report, f, indent=2, ensure_ascii=False)

    logger.info(f"\n报告已保存到: {report_path}")

    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("摘要")
    logger.info("=" * 60)
    logger.info(f"检查文件总数: {missing_report['total_files']}")
    logger.info(f"有缺失值的文件: {missing_report['files_with_missing']} ({missing_report['missing_percentage']:.2f}%)")
    logger.info(f"总数据行数: {missing_report['total_rows']}")

    if missing_report["missing_summary"]:
        logger.info("\n缺失值最多的列:")
        top_missing = sorted(
            missing_report["missing_summary"].items(), key=lambda x: x[1]["files_count"], reverse=True
        )[:5]
        for col, info in top_missing:
            logger.info(f"  - {col}: {info['files_count']}个文件, 平均缺失率 {info['avg_percentage']:.2f}%")
    else:
        logger.info("\n✅ 没有发现缺失值！")


if __name__ == "__main__":
    start_time = datetime.now().timestamp()
    failure_reason = None
    try:
        # 检查数据缺失值
        check_data_quality_summary("data/baostock")
    except Exception as exc:
        failure_reason = str(exc)
        raise
    finally:
        log_task_summary(
            logger,
            task_name="check_missing_values",
            window="data/baostock",
            elapsed_s=datetime.now().timestamp() - start_time,
            error=failure_reason,
        )
