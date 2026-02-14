#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
东方财富 行业研报抓取（行业级）

数据接口来自 reportapi.eastmoney.com/report/list（行业研报列表）
参考：CSDN 示例中公开的请求参数与分页结构。

示例：
python3 scripts/data/policy/fetch_eastmoney_industry_reports.py --begin-date 2024-01-01 --end-date 2024-12-31
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import disable_proxy, get_data_path, setup_logger


logger = setup_logger(Path(__file__).stem, module="data")


@dataclass
class FetchConfig:
    begin_date: str
    end_date: str
    page_size: int = 50
    max_pages: Optional[int] = None
    sleep_seconds: float = 1.0
    industry_code: str = "*"
    rating: str = "*"
    rating_change: str = "*"
    output_dir: Path = get_data_path("raw", "policy", ensure=True)
    state_path: Path = get_data_path("states", ensure=True) / "eastmoney_hyyb.json"


def _jsonp_to_json(text: str) -> Dict:
    text = text.strip()
    if text.endswith(")"):
        left = text.find("(")
        if left != -1:
            text = text[left + 1:-1]
    return json.loads(text)


def _build_url(cfg: FetchConfig, page_no: int) -> str:
    cb = f"datatable{random.randint(1000000, 9999999)}"
    params = {
        "cb": cb,
        "industryCode": cfg.industry_code,
        "pageSize": cfg.page_size,
        "industry": "*",
        "rating": cfg.rating,
        "ratingChange": cfg.rating_change,
        "beginTime": cfg.begin_date,
        "endTime": cfg.end_date,
        "pageNo": page_no,
        "fields": "",
        "qType": "1",
        "orgCode": "",
        "code": "*",
        "rcode": "",
        "_": str(int(time.time() * 1000)),
    }
    return "https://reportapi.eastmoney.com/report/list?" + urllib.parse.urlencode(params)


def _fetch_json(url: str, timeout: int = 20) -> Dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "Referer": "https://data.eastmoney.com/report/hyyb.html",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode(resp.headers.get_content_charset() or "utf-8", errors="ignore")
    return _jsonp_to_json(raw)


def _load_state(path: Path) -> int:
    if not path.exists():
        return 1
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("page_no", 1))
    except Exception:
        return 1


def _save_state(path: Path, page_no: int, meta: Dict) -> None:
    payload = {
        "page_no": page_no,
        "meta": meta,
        "updated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_rows(rows: List[Dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # 兼容字段命名
    title_col = "title" if "title" in df.columns else "Title" if "Title" in df.columns else None
    if title_col:
        df["title"] = df[title_col].astype(str)
    date_col = None
    for cand in ["publishDate", "PublishDate", "pubDate", "pubdate", "reportDate"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col:
        df["publish_date"] = pd.to_datetime(df[date_col], errors="coerce")
    org_col = None
    for cand in ["orgSName", "orgName", "OrgName"]:
        if cand in df.columns:
            org_col = cand
            break
    if org_col:
        df["org"] = df[org_col].astype(str)
    ind_col = None
    for cand in ["industryName", "IndustryName", "industry", "Industry"]:
        if cand in df.columns:
            ind_col = cand
            break
    if ind_col:
        df["industry"] = df[ind_col].astype(str)
    rating_col = None
    for cand in ["rating", "Rating", "ratingName"]:
        if cand in df.columns:
            rating_col = cand
            break
    if rating_col:
        df["rating"] = df[rating_col].astype(str)
    rating_change_col = None
    for cand in ["ratingChange", "RatingChange"]:
        if cand in df.columns:
            rating_change_col = cand
            break
    if rating_change_col:
        df["rating_change"] = df[rating_change_col].astype(str)
    info_col = None
    for cand in ["infoCode", "InfoCode", "reportId", "ReportID"]:
        if cand in df.columns:
            info_col = cand
            break
    if info_col:
        df["info_code"] = df[info_col].astype(str)
    df["source_name"] = "东方财富行业研报"
    df["source_type"] = "research_report"
    df["content"] = (
        df.get("title", "")
        + " "
        + df.get("industry", "")
        + " "
        + df.get("org", "")
        + " "
        + df.get("rating", "")
    )
    df = df.dropna(subset=["publish_date", "title"])
    return df


def fetch_reports(cfg: FetchConfig, resume: bool = True) -> pd.DataFrame:
    page_no = _load_state(cfg.state_path) if resume else 1
    all_rows: List[Dict] = []
    total_pages = None

    while True:
        url = _build_url(cfg, page_no)
        data = _fetch_json(url)
        rows = data.get("data") or []
        if not rows:
            break
        all_rows.extend(rows)
        total_pages = data.get("TotalPage") or data.get("totalPage")
        _save_state(cfg.state_path, page_no, {"total_pages": total_pages, "url": url})
        logger.info(f"已获取第 {page_no} 页, 行数 {len(rows)}")
        page_no += 1
        if cfg.max_pages and page_no > cfg.max_pages:
            break
        if total_pages and page_no > int(total_pages):
            break
        time.sleep(cfg.sleep_seconds)

    return _normalize_rows(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--begin-date", required=True, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--industry-code", type=str, default="*")
    parser.add_argument("--rating", type=str, default="*")
    parser.add_argument("--rating-change", type=str, default="*")
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("raw", "policy", ensure=True)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = FetchConfig(
        begin_date=args.begin_date,
        end_date=args.end_date,
        page_size=args.page_size,
        max_pages=args.max_pages,
        sleep_seconds=args.sleep_seconds,
        industry_code=args.industry_code,
        rating=args.rating,
        rating_change=args.rating_change,
        output_dir=output_dir,
    )

    disable_proxy()
    df = fetch_reports(cfg, resume=args.resume)
    if df.empty:
        logger.warning("未获取到行业研报")
        return
    output_path = output_dir / "eastmoney_industry_reports.parquet"
    if output_path.exists():
        old = pd.read_parquet(output_path)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["publish_date", "title", "org", "industry"])
    df.to_parquet(output_path, index=False)
    logger.info(f"行业研报已保存: {output_path}")


if __name__ == "__main__":
    main()
