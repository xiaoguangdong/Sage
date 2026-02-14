from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from scripts.data._shared.tushare_helpers import PagedDownloader, get_pro, request_with_retry
from scripts.data.macro.paths import CONCEPTS_DIR


def run_dc_index(
    start_date: str = "20200101",
    end_date: Optional[str] = None,
    sleep_seconds: int = 40,
    output_dir: Optional[Path] = None,
) -> None:
    pro = get_pro()
    output_dir = output_dir or Path(CONCEPTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("获取概念指数历史数据 (dc_index)")
    print("=" * 80)

    output_file = output_dir / "dc_index.parquet"
    progress_file = output_dir / "dc_index_progress.txt"

    if progress_file.exists():
        last_date = progress_file.read_text(encoding="utf-8").strip()
        if last_date:
            print(f"找到进度文件，上次获取到: {last_date}")
            start_date = str(int(last_date) + 1)
    else:
        print(f"无进度文件，从 {start_date} 开始")

    existing_data = pd.read_parquet(output_file) if output_file.exists() else pd.DataFrame()

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.now() - timedelta(days=1) if end_date is None else datetime.strptime(end_date, "%Y%m%d")
    months = []
    current_dt = datetime(start_dt.year, start_dt.month, 1)
    while current_dt <= end_dt:
        months.append(current_dt)
        if current_dt.month == 12:
            current_dt = datetime(current_dt.year + 1, 1, 1)
        else:
            current_dt = datetime(current_dt.year, current_dt.month + 1, 1)

    print(f"需要获取 {len(months)} 个月的概念指数数据")

    all_index = []
    for i, month_start in enumerate(months, start=1):
        month_end = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        if month_end > end_dt:
            month_end = end_dt
        start_str = month_start.strftime("%Y%m%d")
        end_str = month_end.strftime("%Y%m%d")

        print(f"\n[{i}/{len(months)}] 获取 {start_str} ~ {end_str} 的概念指数数据...")
        downloader = PagedDownloader(pro, "dc_index", limit=5000, sleep_seconds=sleep_seconds)
        month_df = downloader.fetch_pages({"start_date": start_str, "end_date": end_str})
        if month_df is None or month_df.empty:
            print(f"  {start_str} ~ {end_str} 未获取到数据")
            continue

        all_index.append(month_df)
        if len(all_index) >= 6:
            new_data = pd.concat(all_index, ignore_index=True)
            combined = pd.concat([existing_data, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset=["trade_date", "ts_code"])
            combined.to_parquet(output_file, index=False)
            print(f"  已保存到 {output_file}")
            existing_data = combined
            all_index = []

        progress_file.write_text(end_str, encoding="utf-8")

    if all_index:
        new_data = pd.concat(all_index, ignore_index=True)
        combined = pd.concat([existing_data, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["trade_date", "ts_code"])
        combined.to_parquet(output_file, index=False)
        print(f"\n已保存到 {output_file}")

    if output_file.exists():
        final_df = pd.read_parquet(output_file)
        print(f"\n总记录数: {len(final_df)}")
        if len(final_df) > 0:
            print(f"日期范围: {final_df['trade_date'].min()} ~ {final_df['trade_date'].max()}")
            print(f"概念数量: {final_df['ts_code'].nunique()}")


def run_ths_index(output_dir: Optional[Path] = None) -> None:
    pro = get_pro()
    output_dir = output_dir or Path(CONCEPTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("获取同花顺板块指数列表 (ths_index)")
    print("=" * 80)

    output_file = output_dir / "ths_index.parquet"
    df = request_with_retry(pro, "ths_index", {}, max_retries=3, sleep_seconds=1)
    if df is None or df.empty:
        print("  获取失败或无数据")
        return
    df.to_parquet(output_file, index=False)
    print(f"  已保存到 {output_file}，行数 {len(df)}")


def run_ths_daily(
    start_date: str = "20200101",
    end_date: Optional[str] = None,
    sleep_seconds: int = 40,
    all_by_month: bool = False,
    output_dir: Optional[Path] = None,
) -> None:
    pro = get_pro()
    output_dir = output_dir or Path(CONCEPTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("获取同花顺板块指数行情 (ths_daily)")
    print("=" * 80)

    output_file = output_dir / "ths_daily.parquet"
    progress_file = output_dir / "ths_daily_progress.txt"
    last_code = ""
    last_month = ""
    if progress_file.exists():
        content = progress_file.read_text(encoding="utf-8").strip()
        if content:
            parts = content.split(",")
            last_code = parts[0]
            last_month = parts[1] if len(parts) > 1 else ""
        print(f"  找到进度文件，上次到: {last_code} {last_month}")

    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.now() - timedelta(days=1) if end_date is None else datetime.strptime(end_date, "%Y%m%d")
    months = []
    current_dt = datetime(start_dt.year, start_dt.month, 1)
    while current_dt <= end_dt:
        months.append(current_dt)
        if current_dt.month == 12:
            current_dt = datetime(current_dt.year + 1, 1, 1)
        else:
            current_dt = datetime(current_dt.year, current_dt.month + 1, 1)

    existing = pd.read_parquet(output_file) if output_file.exists() else pd.DataFrame()

    if all_by_month:
        for i, month_start in enumerate(months, start=1):
            month_end = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            if month_end > end_dt:
                month_end = end_dt
            start_str = month_start.strftime("%Y%m%d")
            end_str = month_end.strftime("%Y%m%d")

            if last_month and end_str <= last_month:
                continue

            print(f"[{i}/{len(months)}] 获取 {start_str} ~ {end_str} 全量行情...")
            downloader = PagedDownloader(pro, "ths_daily", limit=3000, sleep_seconds=sleep_seconds)
            df = downloader.fetch_pages({"start_date": start_str, "end_date": end_str})
            if df is not None and not df.empty:
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["ts_code", "trade_date"])
                combined.to_parquet(output_file, index=False)
                existing = combined
            progress_file.write_text(f"ALL,{end_str}", encoding="utf-8")
        return

    index_file = output_dir / "ths_index.parquet"
    if not index_file.exists():
        print("  缺少 ths_index，请先执行 ths_index 拉取")
        return

    index_df = pd.read_parquet(index_file)
    if "ts_code" not in index_df.columns:
        print("  ths_index 缺少 ts_code 字段")
        return

    codes = index_df["ts_code"].dropna().astype(str).unique().tolist()
    if last_code and last_code in codes:
        start_idx = codes.index(last_code) + 1
    else:
        start_idx = 0

    for i, code in enumerate(codes[start_idx:], start=start_idx):
        print(f"[{i+1}/{len(codes)}] 获取 {code} 行情...")
        for month_start in months:
            month_end = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            if month_end > end_dt:
                month_end = end_dt
            start_str = month_start.strftime("%Y%m%d")
            end_str = month_end.strftime("%Y%m%d")

            if code == last_code and last_month and end_str <= last_month:
                continue

            df = request_with_retry(
                pro,
                "ths_daily",
                {"ts_code": code, "start_date": start_str, "end_date": end_str},
                max_retries=3,
                sleep_seconds=sleep_seconds,
            )
            if df is not None and not df.empty:
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["ts_code", "trade_date"])
                combined.to_parquet(output_file, index=False)
                existing = combined
            progress_file.write_text(f"{code},{end_str}", encoding="utf-8")


def run_concept_data_full(
    start_date: str = "20200101",
    end_date: Optional[str] = None,
    sleep_seconds: int = 40,
    output_dir: Optional[Path] = None,
) -> None:
    """获取概念指数/成分/日线（较完整版本）"""
    pro = get_pro()
    output_dir = output_dir or Path(CONCEPTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_with_paging(api_name: str, params: dict, limit: int, progress_path: Path, key: str):
        offset = 0
        page = 1
        frames = []
        while True:
            print(f"  第{page}页获取 (offset={offset})...", end=" ")
            df = request_with_retry(
                pro,
                api_name,
                {**params, "offset": offset},
                max_retries=3,
                sleep_seconds=sleep_seconds,
            )
            if df is not None and not df.empty:
                print(f"成功获取 {len(df)} 条")
                frames.append(df)
                offset += len(df)
                page += 1
                if len(df) < limit:
                    break
            else:
                print("无数据或获取失败")
                break
        if frames:
            progress_path.write_text(key, encoding="utf-8")
            return pd.concat(frames, ignore_index=True)
        return None

    print("\n" + "=" * 80)
    print("步骤1: 获取概念指数历史数据")
    print("=" * 80)
    index_file = output_dir / "dc_index.parquet"
    index_progress = output_dir / "dc_index_progress.txt"
    df_index = _fetch_with_paging(
        "dc_index",
        {"start_date": start_date, "end_date": end_date or datetime.now().strftime("%Y%m%d")},
        limit=5000,
        progress_path=index_progress,
        key=(end_date or datetime.now().strftime("%Y%m%d")),
    )
    if df_index is not None and not df_index.empty:
        df_index.to_parquet(index_file, index=False)
        print(f"  已保存到 {index_file}")

    print("\n" + "=" * 80)
    print("步骤2: 获取概念成分股历史数据")
    print("=" * 80)
    member_file = output_dir / "dc_member.parquet"
    member_progress = output_dir / "dc_member_progress.txt"
    df_member = _fetch_with_paging(
        "dc_member",
        {},
        limit=5000,
        progress_path=member_progress,
        key="done",
    )
    if df_member is not None and not df_member.empty:
        df_member.to_parquet(member_file, index=False)
        print(f"  已保存到 {member_file}")

    print("\n" + "=" * 80)
    print("步骤3: 获取概念成分股日线数据")
    print("=" * 80)
    daily_file = output_dir / "dc_daily.parquet"
    daily_progress = output_dir / "dc_daily_progress.txt"
    df_daily = _fetch_with_paging(
        "dc_daily",
        {"start_date": start_date, "end_date": end_date or datetime.now().strftime("%Y%m%d")},
        limit=5000,
        progress_path=daily_progress,
        key=(end_date or datetime.now().strftime("%Y%m%d")),
    )
    if df_daily is not None and not df_daily.empty:
        df_daily.to_parquet(daily_file, index=False)
        print(f"  已保存到 {daily_file}")
