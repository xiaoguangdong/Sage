#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
国家统计局 easyquery 工业品产量数据获取（A020901/A020902）

示例请求（浏览器可用）：
https://data.stats.gov.cn/easyquery.htm?m=QueryData&dbcode=hgyd&rowcode=zb&colcode=sj&wds=%5B%5D&dfwds=%5B%7B%22wdcode%22%3A%22zb%22%2C%22valuecode%22%3A%22A020901%22%7D%5D&k1=...&h=1

用法：
python scripts/data/macro/fetch_nbs_easyquery_output.py --zb-codes A020901,A020902
python scripts/data/macro/fetch_nbs_easyquery_output.py --zb-codes A020901 --sj-code 2020-2025
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import requests

from scripts.data.macro.paths import MACRO_DIR


def build_params(
    zb_code: str, sj_code: Optional[str] = None, dbcode: str = "hgyd", rowcode: str = "zb", colcode: str = "sj"
) -> dict:
    dfwds = [{"wdcode": "zb", "valuecode": zb_code}]
    if sj_code:
        dfwds.append({"wdcode": "sj", "valuecode": sj_code})
    return {
        "m": "QueryData",
        "dbcode": dbcode,
        "rowcode": rowcode,
        "colcode": colcode,
        "wds": "[]",
        "dfwds": json.dumps(dfwds, ensure_ascii=False),
        "k1": str(int(time.time() * 1000)),
        "h": 1,
    }


def safe_filename(zb_code: str, suffix: str = "easyquery") -> str:
    return f"{zb_code}_{suffix}.json"


def build_year_ranges(start_year: Optional[int], end_year: Optional[int], span_years: int = 5) -> List[str]:
    if not start_year or not end_year:
        return []
    if start_year > end_year:
        start_year, end_year = end_year, start_year
    ranges = []
    year = start_year
    while year <= end_year:
        end = min(end_year, year + span_years - 1)
        ranges.append(f"{year}-{end}")
        year = end + 1
    return ranges


def fetch_easyquery_json(
    zb_code: str,
    base_url: str,
    session: requests.Session,
    headers: dict,
    sj_code: Optional[str],
    sleep_seconds: int,
) -> dict:
    params = build_params(zb_code=zb_code, sj_code=sj_code)
    response = session.get(base_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    time.sleep(sleep_seconds)
    return data


def main():
    parser = argparse.ArgumentParser(description="NBS easyquery 工业品产量数据获取")
    parser.add_argument("--zb-codes", default="A020901,A020902", help="指标代码，逗号分隔")
    parser.add_argument("--sj-code", default=None, help="时间范围（如 2020-2025 或 LAST60）")
    parser.add_argument("--start-year", type=int, default=None, help="起始年份（自动生成时间范围）")
    parser.add_argument("--end-year", type=int, default=None, help="结束年份（自动生成时间范围）")
    parser.add_argument("--span-years", type=int, default=5, help="自动分段跨度（年）")
    parser.add_argument("--out-dir", default=str(MACRO_DIR), help="输出目录")
    parser.add_argument("--base-url", default="https://data.stats.gov.cn/easyquery.htm")
    parser.add_argument("--sleep-seconds", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Referer": "https://data.stats.gov.cn/easyquery.htm?cn=C01",
    }

    session = requests.Session()
    zb_codes: List[str] = [c.strip() for c in args.zb_codes.split(",") if c.strip()]

    sj_ranges = build_year_ranges(args.start_year, args.end_year, args.span_years)
    if args.sj_code:
        sj_ranges = [args.sj_code]

    for idx, zb_code in enumerate(zb_codes, start=1):
        print(f"[{idx}/{len(zb_codes)}] 获取 {zb_code} ...")
        if not sj_ranges:
            data = fetch_easyquery_json(
                zb_code=zb_code,
                base_url=args.base_url,
                session=session,
                headers=headers,
                sj_code=args.sj_code,
                sleep_seconds=args.sleep_seconds,
            )
            filename = safe_filename(zb_code)
            out_path = out_dir / filename
            out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            print(f"  已保存: {out_path}")
        else:
            for sj_code in sj_ranges:
                data = fetch_easyquery_json(
                    zb_code=zb_code,
                    base_url=args.base_url,
                    session=session,
                    headers=headers,
                    sj_code=sj_code,
                    sleep_seconds=args.sleep_seconds,
                )
                filename = safe_filename(zb_code, suffix=f"{sj_code}")
                out_path = out_dir / filename
                out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                print(f"  已保存: {out_path}")


if __name__ == "__main__":
    main()
