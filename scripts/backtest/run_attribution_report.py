#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sage_core.backtest.attribution import brinson_attribution, factor_attribution
from scripts.data._shared.runtime import get_data_path, setup_logger

logger = setup_logger("attribution_report", module="backtest")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def run_brinson(
    portfolio_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    date_col: str,
    industry_col: str,
    weight_col: str,
    return_col: str,
) -> None:
    portfolio = _read_table(portfolio_path)
    benchmark = _read_table(benchmark_path)
    result = brinson_attribution(
        portfolio,
        benchmark,
        date_col=date_col,
        industry_col=industry_col,
        weight_col=weight_col,
        return_col=return_col,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.by_industry.to_parquet(output_dir / "brinson_by_industry.parquet", index=False)
    result.by_date.to_parquet(output_dir / "brinson_by_date.parquet", index=False)
    logger.info("Brinson归因已输出: %s", output_dir)


def run_factor(
    exposure_path: Path,
    factor_cols: list[str],
    output_dir: Path,
    date_col: str,
    weight_col: str,
    return_col: str,
) -> None:
    exposures = _read_table(exposure_path)
    result = factor_attribution(
        exposures,
        factor_cols,
        date_col=date_col,
        weight_col=weight_col,
        return_col=return_col,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.factor_returns.to_parquet(output_dir / "factor_returns.parquet", index=False)
    result.factor_contributions.to_parquet(output_dir / "factor_contributions.parquet", index=False)
    logger.info("因子归因已输出: %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="收益归因报告")
    parser.add_argument("--portfolio", type=str, help="组合持仓文件(含权重与收益)")
    parser.add_argument("--benchmark", type=str, help="基准持仓文件(含权重与收益)")
    parser.add_argument("--factor-exposure", type=str, help="因子暴露文件(含收益与权重)")
    parser.add_argument("--factor-cols", type=str, default="", help="因子列名, 逗号分隔")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--date-col", type=str, default="trade_date")
    parser.add_argument("--industry-col", type=str, default="industry_l1")
    parser.add_argument("--weight-col", type=str, default="weight")
    parser.add_argument("--return-col", type=str, default="return")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("backtest", "attribution", ensure=True)

    if args.portfolio and args.benchmark:
        run_brinson(
            Path(args.portfolio),
            Path(args.benchmark),
            output_dir,
            args.date_col,
            args.industry_col,
            args.weight_col,
            args.return_col,
        )

    if args.factor_exposure and args.factor_cols:
        factor_cols = [c.strip() for c in args.factor_cols.split(",") if c.strip()]
        if factor_cols:
            run_factor(
                Path(args.factor_exposure),
                factor_cols,
                output_dir,
                args.date_col,
                args.weight_col,
                args.return_col,
            )


if __name__ == "__main__":
    main()
