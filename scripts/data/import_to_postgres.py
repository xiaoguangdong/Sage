#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将当前数据导入到 PostgreSQL（单库）

默认读取 config/base.yaml 的 data 目录与 env 中的 SAGE_DB_*。

用法示例：
  python scripts/data/import_to_postgres.py --task daily_kline
  python scripts/data/import_to_postgres.py --task all --batch-size 50000
  python scripts/data/import_to_postgres.py --task cn_macro --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pyarrow.dataset as ds
except Exception:  # pragma: no cover - pyarrow 为可选依赖
    ds = None

try:
    import psycopg

    try:
        from psycopg.extras import execute_values
    except Exception:
        execute_values = None
except Exception:  # pragma: no cover - 运行时报错提示
    psycopg = None
    execute_values = None

from scripts.data._shared.runtime import get_tushare_root, setup_logger

logger = setup_logger("import_postgres", module="data")


@dataclass
class DbConfig:
    host: str
    port: int
    name: str
    user: str
    password: str

    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def load_db_config(args: argparse.Namespace) -> DbConfig:
    return DbConfig(
        host=args.db_host or _env("SAGE_DB_HOST", "127.0.0.1"),
        port=int(args.db_port or _env("SAGE_DB_PORT", "5432")),
        name=args.db_name or _env("SAGE_DB_NAME", "sage_db"),
        user=args.db_user or _env("SAGE_DB_USER", "sage"),
        password=args.db_password or _env("SAGE_DB_PASSWORD", "sage_dev_2026"),
    )


def _require_psycopg() -> None:
    if psycopg is None:
        raise RuntimeError("缺少 psycopg 依赖，请先安装：pip install 'psycopg[binary]'")


def _normalize_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    series = df[col]
    if pd.api.types.is_datetime64_any_dtype(series):
        df[col] = pd.to_datetime(series, errors="coerce").dt.date
        return df
    df[col] = pd.to_datetime(series.astype(str), errors="coerce").dt.date
    return df


def _iter_parquet_batches(paths: List[Path], columns: Optional[List[str]], batch_size: int) -> Iterable[pd.DataFrame]:
    if ds is None:
        raise RuntimeError("缺少 pyarrow.dataset，无法分批读取parquet")
    dataset = ds.dataset([str(p) for p in paths], format="parquet")
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        yield batch.to_pandas()


def _read_parquet_all(paths: List[Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _insert_dataframe(
    conn: "psycopg.Connection",
    table: str,
    df: pd.DataFrame,
    columns: List[str],
    batch_size: int,
    dry_run: bool,
) -> int:
    if df.empty:
        return 0
    df = df[columns].copy()
    total = len(df)
    if dry_run:
        logger.info("[dry-run] %s rows=%d", table, total)
        return total

    cols_sql = ", ".join(columns)
    with conn.cursor() as cur:
        for start in range(0, total, batch_size):
            chunk = df.iloc[start : start + batch_size]
            chunk = chunk.astype(object).where(pd.notna(chunk), None)
            rows = list(chunk.itertuples(index=False, name=None))
            if not rows:
                continue
            if execute_values is not None:
                sql = f"INSERT INTO {table} ({cols_sql}) VALUES %s ON CONFLICT DO NOTHING"
                execute_values(cur, sql, rows, page_size=min(batch_size, 5000))
            else:
                placeholders = ", ".join(["%s"] * len(columns))
                sql = f"INSERT INTO {table} ({cols_sql}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
                cur.executemany(sql, rows)
        conn.commit()
    return total


def _load_daily_kline(tushare_root: Path) -> List[Path]:
    daily_dir = tushare_root / "daily"
    if daily_dir.exists():
        files = sorted(daily_dir.glob("daily_*.parquet"))
        if files:
            return files
    single = tushare_root / "daily.parquet"
    return [single] if single.exists() else []


def _load_moneyflow_paths(tushare_root: Path) -> List[Path]:
    mf_dir = tushare_root / "moneyflow"
    if not mf_dir.exists():
        return []
    return sorted(mf_dir.glob("*.parquet"))


def _task_daily_kline(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    return (
        "market.daily_kline",
        _load_daily_kline(tushare_root),
        [
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
        ],
    )


def _task_daily_basic(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "daily_basic.parquet"
    if not path.exists():
        path = tushare_root / "daily_basic_all.parquet"
    return (
        "market.daily_basic",
        [path],
        [
            "ts_code",
            "trade_date",
            "close",
            "turnover_rate",
            "turnover_rate_f",
            "volume_ratio",
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "dv_ratio",
            "dv_ttm",
            "total_share",
            "float_share",
            "free_share",
            "total_mv",
            "circ_mv",
        ],
    )


def _task_index_ohlc(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "index_ohlc_all.parquet"
    return (
        "market.index_ohlc",
        [path],
        [
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount",
        ],
    )


def _task_hs300_constituents(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "constituents" / "hs300_constituents_all.parquet"
    return "market.hs300_constituents", [path], ["index_code", "con_code", "trade_date", "weight"]


def _task_income(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "fundamental" / "income" / "income_all.parquet"
    return (
        "fundamental.income",
        [path],
        [
            "ts_code",
            "ann_date",
            "f_ann_date",
            "end_date",
            "report_type",
            "comp_type",
            "basic_eps",
            "diluted_eps",
            "total_revenue",
            "revenue",
            "total_cogs",
            "oper_cost",
            "sell_exp",
            "admin_exp",
            "rd_exp",
            "fin_exp",
            "operate_profit",
            "non_oper_income",
            "non_oper_exp",
            "n_income",
            "n_income_attr_p",
            "ebit",
            "ebitda",
        ],
    )


def _task_balancesheet(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "fundamental" / "balancesheet" / "balancesheet_all.parquet"
    return (
        "fundamental.balancesheet",
        [path],
        [
            "ts_code",
            "ann_date",
            "f_ann_date",
            "end_date",
            "report_type",
            "comp_type",
            "total_assets",
            "total_liab",
            "total_hldr_eqy_exc_min_int",
            "total_cur_assets",
            "total_nca",
            "total_cur_liab",
            "total_ncl",
            "money_cap",
            "inventories",
            "accounts_receiv",
            "fix_assets",
            "intan_assets",
            "goodwill",
            "lt_borr",
            "st_borr",
        ],
    )


def _task_cashflow(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "fundamental" / "cashflow" / "cashflow_all.parquet"
    return (
        "fundamental.cashflow",
        [path],
        [
            "ts_code",
            "ann_date",
            "f_ann_date",
            "end_date",
            "report_type",
            "comp_type",
            "n_cashflow_act",
            "n_cashflow_inv_act",
            "n_cash_flows_fnc_act",
            "c_fr_sale_sg",
            "c_paid_goods_s",
            "c_paid_to_for_empl",
            "c_pay_acq_const_fiolta",
            "free_cashflow",
        ],
    )


def _task_fina_indicator(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "fina_indicator" / "fina_indicator_all.parquet"
    return (
        "fundamental.fina_indicator",
        [path],
        [
            "ts_code",
            "ann_date",
            "end_date",
            "roe",
            "roe_dt",
            "roa",
            "grossprofit_margin",
            "netprofit_margin",
            "debt_to_assets",
            "current_ratio",
            "quick_ratio",
            "eps",
            "bps",
            "cfps",
            "ocfps",
            "or_yoy",
            "op_yoy",
            "netprofit_yoy",
        ],
    )


def _task_dividend(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "fundamental" / "dividend" / "dividend_all.parquet"
    return (
        "fundamental.dividend",
        [path],
        [
            "ts_code",
            "end_date",
            "ann_date",
            "div_proc",
            "stk_div",
            "stk_bo_rate",
            "stk_co_rate",
            "cash_div",
            "cash_div_tax",
            "record_date",
            "ex_date",
            "pay_date",
            "div_listdate",
        ],
    )


def _task_northbound_flow(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "northbound" / "northbound_flow.parquet"
    return (
        "flow.northbound_flow",
        [path],
        [
            "trade_date",
            "ggt_ss",
            "ggt_sz",
            "hgt",
            "sgt",
            "north_money",
            "south_money",
        ],
    )


def _task_northbound_hold(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "northbound" / "northbound_hold.parquet"
    return (
        "flow.northbound_hold",
        [path],
        [
            "ts_code",
            "trade_date",
            "name",
            "vol",
            "ratio",
            "exchange",
        ],
    )


def _task_northbound_top10(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "northbound" / "hsgt_top10.parquet"
    return (
        "flow.northbound_top10",
        [path],
        [
            "trade_date",
            "ts_code",
            "name",
            "close",
            "change",
            "rank",
            "market_type",
            "amount",
            "net_amount",
            "buy",
            "sell",
        ],
    )


def _task_margin(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "margin.parquet"
    return (
        "flow.margin",
        [path],
        [
            "trade_date",
            "exchange_id",
            "rzye",
            "rzmre",
            "rzche",
            "rqye",
            "rqmcl",
            "rzrqye",
            "rqylje",
        ],
    )


def _task_moneyflow(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    return (
        "flow.moneyflow",
        _load_moneyflow_paths(tushare_root),
        [
            "ts_code",
            "trade_date",
            "buy_sm_vol",
            "buy_sm_amount",
            "sell_sm_vol",
            "sell_sm_amount",
            "buy_md_vol",
            "buy_md_amount",
            "sell_md_vol",
            "sell_md_amount",
            "buy_lg_vol",
            "buy_lg_amount",
            "sell_lg_vol",
            "sell_lg_amount",
            "buy_elg_vol",
            "buy_elg_amount",
            "sell_elg_vol",
            "sell_elg_amount",
            "net_mf_vol",
            "net_mf_amount",
        ],
    )


def _task_ths_index(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "concepts" / "ths_index.parquet"
    return "concept.ths_index", [path], ["ts_code", "name", "count", "exchange", "list_date", "type"]


def _task_ths_daily(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "concepts" / "ths_daily.parquet"
    return (
        "concept.ths_daily",
        [path],
        [
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pct_change",
            "vol",
            "turnover_rate",
            "total_mv",
            "float_mv",
            "pe",
        ],
    )


def _task_ths_member(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "concepts" / "ths_member.parquet"
    return "concept.ths_member", [path], ["ts_code", "con_code", "name"]


def _task_sw_index_member(tushare_root: Path) -> tuple[str, List[Path], List[str]]:
    path = tushare_root / "sw_industry" / "sw_index_member.parquet"
    return (
        "concept.sw_index_member",
        [path],
        ["index_code", "index_name", "con_code", "con_name", "in_date", "out_date"],
    )


def _task_yield_curve(tushare_root: Path, curve_type: str) -> pd.DataFrame:
    frames = []
    for term in (10, 2):
        path = tushare_root / "macro" / f"yield_{term}y.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["curve_term"] = float(term)
        df["curve_type"] = curve_type
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), errors="coerce").dt.date
    return df[["trade_date", "curve_type", "curve_term", "yield"]]


def _task_cn_macro(tushare_root: Path) -> pd.DataFrame:
    frames = []
    cpi = tushare_root / "macro" / "tushare_cpi.parquet"
    if cpi.exists():
        df = pd.read_parquet(cpi)
        df = df[["month", "nt_val", "nt_yoy", "nt_mom", "nt_accu"]].copy()
        df["indicator"] = "cpi"
        frames.append(df)

    ppi = tushare_root / "macro" / "tushare_ppi.parquet"
    if ppi.exists():
        df = pd.read_parquet(ppi)
        df = df.copy()
        df["nt_val"] = df["ppi_yoy"]
        df["nt_yoy"] = df["ppi_yoy"]
        df["nt_mom"] = df["ppi_mom"]
        df["nt_accu"] = df["ppi_accu"]
        df["indicator"] = "ppi"
        df = df[["month", "indicator", "nt_val", "nt_yoy", "nt_mom", "nt_accu"]]
        frames.append(df)

    pmi = tushare_root / "macro" / "tushare_pmi.parquet"
    if pmi.exists():
        df = pd.read_parquet(pmi)
        month_col = "MONTH" if "MONTH" in df.columns else "month" if "month" in df.columns else None
        if month_col and "PMI010000" in df.columns:
            pmi_df = df[[month_col, "PMI010000"]].copy()
            pmi_df.rename(columns={month_col: "month", "PMI010000": "nt_val"}, inplace=True)
            pmi_df["indicator"] = "pmi"
            pmi_df["nt_yoy"] = pd.NA
            pmi_df["nt_mom"] = pd.NA
            pmi_df["nt_accu"] = pd.NA
            pmi_df = pmi_df[["month", "indicator", "nt_val", "nt_yoy", "nt_mom", "nt_accu"]]
            frames.append(pmi_df)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["month"] = df["month"].astype(str)
    return df


TASK_BUILDERS = {
    "daily_kline": _task_daily_kline,
    "daily_basic": _task_daily_basic,
    "index_ohlc": _task_index_ohlc,
    "hs300_constituents": _task_hs300_constituents,
    "income": _task_income,
    "balancesheet": _task_balancesheet,
    "cashflow": _task_cashflow,
    "fina_indicator": _task_fina_indicator,
    "dividend": _task_dividend,
    "northbound_flow": _task_northbound_flow,
    "northbound_hold": _task_northbound_hold,
    "northbound_top10": _task_northbound_top10,
    "margin": _task_margin,
    "moneyflow": _task_moneyflow,
    "ths_index": _task_ths_index,
    "ths_daily": _task_ths_daily,
    "ths_member": _task_ths_member,
    "sw_index_member": _task_sw_index_member,
}


def _apply_task_transforms(task: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if task in {"daily_kline", "daily_basic", "index_ohlc"}:
        if "code" in df.columns and "ts_code" not in df.columns:
            df["ts_code"] = df["code"]
        if "date" in df.columns and "trade_date" not in df.columns:
            df["trade_date"] = df["date"]
        if "pct_change" in df.columns and "pct_chg" not in df.columns:
            df["pct_chg"] = df["pct_change"]
        df = _normalize_date_col(df, "trade_date")
    if task in {"income", "balancesheet", "cashflow", "fina_indicator"}:
        df = _normalize_date_col(df, "ann_date")
        df = _normalize_date_col(df, "f_ann_date")
        df = _normalize_date_col(df, "end_date")
    if task == "dividend":
        df = _normalize_date_col(df, "ann_date")
        df = _normalize_date_col(df, "end_date")
        df = _normalize_date_col(df, "record_date")
        df = _normalize_date_col(df, "ex_date")
        df = _normalize_date_col(df, "pay_date")
        df = _normalize_date_col(df, "div_listdate")
    if task in {"northbound_flow", "northbound_hold", "northbound_top10", "margin", "moneyflow"}:
        df = _normalize_date_col(df, "trade_date")
    if task == "margin":
        if "rqyl" in df.columns and "rqylje" not in df.columns:
            df["rqylje"] = df["rqyl"]
    if task == "northbound_hold":
        if "code" in df.columns:
            df = df.drop(columns=["code"], errors="ignore")
    if task == "ths_daily":
        df = _normalize_date_col(df, "trade_date")
        for col in ["total_mv", "float_mv", "pe"]:
            if col not in df.columns:
                df[col] = pd.NA
    if task == "ths_index":
        df = _normalize_date_col(df, "list_date")
        if "count" in df.columns:
            df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("Int64")
    if task == "ths_member":
        if "con_name" in df.columns and "name" not in df.columns:
            df["name"] = df["con_name"]
    if task == "sw_index_member":
        df = _normalize_date_col(df, "in_date")
        df = _normalize_date_col(df, "out_date")
        if "index_name" not in df.columns:
            df["index_name"] = pd.NA
        if "con_name" not in df.columns:
            df["con_name"] = pd.NA
    return df


def run_task(
    conn: "psycopg.Connection",
    task: str,
    tushare_root: Path,
    batch_size: int,
    dry_run: bool,
    curve_type: str,
) -> int:
    if task == "yield_curve":
        df = _task_yield_curve(tushare_root, curve_type)
        if df.empty:
            logger.warning("yield_curve 无可用数据")
            return 0
        return _insert_dataframe(
            conn, "macro.yield_curve", df, ["trade_date", "curve_type", "curve_term", "yield"], batch_size, dry_run
        )

    if task == "cn_macro":
        df = _task_cn_macro(tushare_root)
        if df.empty:
            logger.warning("cn_macro 无可用数据")
            return 0
        return _insert_dataframe(
            conn,
            "macro.cn_macro",
            df,
            ["month", "indicator", "nt_val", "nt_yoy", "nt_mom", "nt_accu"],
            batch_size,
            dry_run,
        )

    builder = TASK_BUILDERS.get(task)
    if not builder:
        raise ValueError(f"未知任务: {task}")
    table, paths, columns = builder(tushare_root)
    paths = [p for p in paths if p and p.exists()]
    if not paths:
        logger.warning("%s 未找到数据文件", task)
        return 0

    total_inserted = 0
    if ds and len(paths) > 0:
        for df in _iter_parquet_batches(paths, None, batch_size=batch_size):
            df = _apply_task_transforms(task, df)
            if df.empty:
                continue
            missing_cols = [c for c in columns if c not in df.columns]
            if missing_cols:
                for col in missing_cols:
                    df[col] = pd.NA
            df = df[columns]
            total_inserted += _insert_dataframe(conn, table, df, columns, batch_size, dry_run)
    else:
        df = _read_parquet_all(paths)
        df = _apply_task_transforms(task, df)
        if not df.empty:
            missing_cols = [c for c in columns if c not in df.columns]
            for col in missing_cols:
                df[col] = pd.NA
            df = df[columns]
            total_inserted += _insert_dataframe(conn, table, df, columns, batch_size, dry_run)
    logger.info("%s 导入完成: %d", task, total_inserted)
    return total_inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="导入数据到 Postgres 单库")
    parser.add_argument("--task", required=True, help="任务名或all")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--curve-type", type=str, default="yc_cb")
    parser.add_argument("--db-host", type=str, default=None)
    parser.add_argument("--db-port", type=str, default=None)
    parser.add_argument("--db-name", type=str, default=None)
    parser.add_argument("--db-user", type=str, default=None)
    parser.add_argument("--db-password", type=str, default=None)
    args = parser.parse_args()

    _require_psycopg()
    db_cfg = load_db_config(args)
    tushare_root = get_tushare_root()

    tasks = [args.task]
    if args.task == "all":
        tasks = [
            "daily_kline",
            "daily_basic",
            "index_ohlc",
            "hs300_constituents",
            "income",
            "balancesheet",
            "cashflow",
            "fina_indicator",
            "dividend",
            "northbound_flow",
            "northbound_hold",
            "northbound_top10",
            "margin",
            "moneyflow",
            "ths_index",
            "ths_daily",
            "ths_member",
            "sw_index_member",
            "yield_curve",
            "cn_macro",
        ]

    logger.info("连接数据库: %s", db_cfg.dsn().replace(db_cfg.password, "******"))
    with psycopg.connect(db_cfg.dsn()) as conn:
        for task in tasks:
            run_task(conn, task, tushare_root, args.batch_size, args.dry_run, args.curve_type)


if __name__ == "__main__":
    main()
