#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同花顺个股页 - 机构评级/研报摘要/业绩预告抓取

示例：
python3 scripts/data/policy/fetch_10jqka_reports.py --symbol 002988
python3 scripts/data/policy/fetch_10jqka_reports.py --symbol 002988 --section forecast
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import disable_proxy, get_data_path, setup_logger

logger = setup_logger(Path(__file__).stem, module="data")

RATING_KEYWORDS = [
    "强烈推荐",
    "买入",
    "增持",
    "持有",
    "减持",
    "卖出",
    "中性",
    "推荐",
    "跑赢行业",
    "跑赢大市",
    "优于大市",
]

ORG_KEYWORDS = ["证券", "研究院", "有限公司", "股份", "投资", "银行", "基金", "资产"]
FORECAST_KEYWORDS = [
    "业绩预告",
    "业绩预增",
    "业绩预减",
    "预增",
    "预减",
    "扭亏",
    "首亏",
    "续亏",
    "略增",
    "略减",
    "预盈",
    "预亏",
]

DATE_PATTERNS = [
    re.compile(r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})"),
    re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日"),
]


def _normalize_date(text: str) -> Optional[str]:
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            year, month, day = match.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
    return None


def _strip_tags(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()


def _pick_rating(text: str) -> Optional[str]:
    for keyword in RATING_KEYWORDS:
        if keyword in text:
            return keyword
    return None


def _pick_org(text: str) -> Optional[str]:
    for keyword in ORG_KEYWORDS:
        if keyword in text:
            return text
    return None


def _extract_title(block: str) -> Optional[str]:
    for match in re.finditer(r"<a[^>]*>(.*?)</a>", block, flags=re.S | re.I):
        title = _strip_tags(match.group(1))
        if len(title) >= 6:
            return title
    return None


def _extract_blocks(html: str) -> List[str]:
    blocks: List[str] = []
    for tag in ["tr", "li", "div"]:
        blocks.extend(re.findall(rf"<{tag}[^>]*>.*?</{tag}>", html, flags=re.S | re.I))
    return blocks


def parse_reports(html: str, section_key: Optional[str] = None) -> List[Dict[str, str]]:
    section = html
    if section_key:
        idx = html.find(section_key)
        if idx != -1:
            section = html[max(0, idx - 20000) : idx + 40000]
    else:
        if "stockreport" in html:
            idx = html.find("stockreport")
            section = html[max(0, idx - 20000) : idx + 40000]
        elif "机构评级" in html:
            idx = html.find("机构评级")
            section = html[max(0, idx - 20000) : idx + 40000]

    rows: List[Dict[str, str]] = []
    for block in _extract_blocks(section):
        text = _strip_tags(block)
        if not text:
            continue
        date = _normalize_date(text)
        rating = _pick_rating(text)
        title = _extract_title(block) or (text if len(text) <= 120 else text[:120])
        org = None
        for piece in re.split(r"\s+", text):
            if _pick_org(piece):
                org = piece
                break
        if date and rating:
            rows.append(
                {
                    "publish_date": date,
                    "title": title,
                    "content": text,
                    "rating": rating,
                    "org": org or "",
                    "report_type": "rating",
                }
            )
        elif date and any(k in text for k in FORECAST_KEYWORDS):
            signal = next((k for k in FORECAST_KEYWORDS if k in text), "")
            rows.append(
                {
                    "publish_date": date,
                    "title": title,
                    "content": text,
                    "rating": signal or "业绩预告",
                    "org": org or "",
                    "report_type": "forecast",
                }
            )
    return rows


def _fetch_html(url: str, user_agent: str, timeout: int) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, help="股票代码（如 002988）")
    parser.add_argument("--url", default=None, help="自定义页面URL")
    parser.add_argument("--section", choices=["stockreport", "forecast"], default="stockreport")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--dump-html", action="store_true")
    args = parser.parse_args()

    symbol = args.symbol.strip()
    default_path = "worth/#stockreport" if args.section == "stockreport" else "worth/#forecast"
    url = args.url or f"https://stockpage.10jqka.com.cn/{symbol}/{default_path}"
    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("raw", "policy", ensure=True)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    disable_proxy()
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    html = _fetch_html(url, user_agent, args.timeout)

    if args.dump_html:
        dump_path = output_dir / f"10jqka_{symbol}.html"
        dump_path.write_text(html, encoding="utf-8", errors="ignore")
        logger.info(f"已保存HTML: {dump_path}")

    rows = parse_reports(html, section_key=args.section)
    if not rows:
        logger.warning("未解析到机构评级条目")
        return

    df = pd.DataFrame(rows)
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df = df.dropna(subset=["publish_date"])
    df["symbol"] = symbol
    df["source_name"] = "同花顺"
    df["source_type"] = "research_report"
    df["url"] = url

    output_name = "10jqka_reports.parquet" if args.section == "stockreport" else "10jqka_forecast.parquet"
    output_path = output_dir / output_name
    if output_path.exists():
        old = pd.read_parquet(output_path)
        df = pd.concat([old, df], ignore_index=True)
        df = df.drop_duplicates(subset=["publish_date", "title", "symbol", "org"])
    df.to_parquet(output_path, index=False)
    logger.info(f"研报摘要已保存: {output_path}")


if __name__ == "__main__":
    main()
