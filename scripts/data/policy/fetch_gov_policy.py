#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政府政策文本拉取（RSS/Atom）

支持：
1) 国家发展改革委 / 工信部 / 国务院 / 部委 RSS 或 Atom

输出：
- data/raw/policy/gov_notices.parquet
"""
from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
import time
import ssl
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import disable_proxy, get_data_path, setup_logger


logger = setup_logger(Path(__file__).stem, module="data")


@dataclass
class SourceConfig:
    name: str
    url: str
    source_type: str = "rss"
    tag: Optional[str] = None
    base_url: Optional[str] = None
    insecure: bool = False


@dataclass
class FetchConfig:
    output_dir: Path
    timeout: int = 20
    sleep_seconds: float = 1.0
    user_agent: str = "SagePolicyBot/0.1"
    max_items_per_source: Optional[int] = None
    dump_html: bool = False
    dump_dir: Optional[Path] = None


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _decode_bytes(raw: bytes, charset: Optional[str]) -> str:
    if charset:
        try:
            return raw.decode(charset, errors="ignore")
        except Exception:
            pass
    for enc in ["utf-8", "gb18030", "gbk"]:
        try:
            return raw.decode(enc, errors="ignore")
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def _fetch_url(url: str, cfg: FetchConfig, insecure: bool = False) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": cfg.user_agent})
    context = None
    if insecure:
        context = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=cfg.timeout, context=context) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset()
        return _decode_bytes(raw, charset)


def _child_text(elem: ET.Element, names: List[str]) -> str:
    for child in elem:
        tag = child.tag.split("}")[-1].lower()
        if tag in names:
            if tag == "link" and child.attrib.get("href"):
                return child.attrib.get("href", "").strip()
            if child.text:
                return child.text.strip()
    return ""


def _extract_entry(entry: ET.Element) -> Dict[str, str]:
    title = _child_text(entry, ["title"])
    link = _child_text(entry, ["link"])
    if not link:
        for child in entry:
            tag = child.tag.split("}")[-1].lower()
            if tag == "link" and child.attrib.get("href"):
                link = child.attrib.get("href", "").strip()
                break
    publish_date = _child_text(entry, ["pubdate", "published", "updated", "date"])
    content = _child_text(entry, ["description", "summary", "content", "encoded"])
    return {
        "title": title,
        "url": link,
        "publish_date": publish_date,
        "content": content,
    }


def parse_rss(content: str) -> List[Dict[str, str]]:
    try:
        root = ET.fromstring(content)
    except Exception as exc:
        logger.warning(f"解析失败: {exc}")
        return []

    items: List[Dict[str, str]] = []
    for item in root.findall(".//item"):
        items.append(_extract_entry(item))
    if items:
        return items

    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        items.append(_extract_entry(entry))
    if items:
        return items

    for entry in root.findall(".//entry"):
        items.append(_extract_entry(entry))
    return items


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tokens: List[Dict[str, str]] = []
        self._in_anchor = False
        self._anchor_href = ""
        self._anchor_text: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._in_anchor = True
            self._anchor_href = ""
            self._anchor_text = []
            for key, value in attrs:
                if key.lower() == "href" and value:
                    self._anchor_href = value

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._in_anchor:
            text = "".join(self._anchor_text).strip()
            if text:
                self.tokens.append({
                    "type": "a",
                    "text": text,
                    "href": self._anchor_href,
                })
            self._in_anchor = False
            self._anchor_href = ""
            self._anchor_text = []

    def handle_data(self, data):
        text = (data or "").strip()
        if not text:
            return
        if self._in_anchor:
            self._anchor_text.append(text)
        else:
            self.tokens.append({
                "type": "text",
                "text": text,
            })


_DATE_PATTERN = re.compile(r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})")
_DATE_PATTERN_CN = re.compile(r"(20\d{2})年(\d{1,2})月(\d{1,2})日")
_DATE_PATTERN_MD = re.compile(r"(?:\[)?(\d{1,2})-(\d{1,2})(?:\])?")
_IGNORE_TITLES = {
    "查看详细",
    "查看更多",
    "更多",
    "上一页",
    "下一页",
    "返回",
    ">>",
    "更多>>",
    "首页",
    "新闻",
    "新闻报道",
    "新闻发布",
    "沟通交流",
    "高级搜索",
}


def _normalize_date(raw: str) -> Optional[str]:
    match = _DATE_PATTERN.search(raw)
    if not match:
        match = _DATE_PATTERN_CN.search(raw)
        if not match:
            match = _DATE_PATTERN_MD.search(raw)
            if not match:
                return None
            month, day = match.groups()
            today = pd.Timestamp.today()
            month_i = int(month)
            day_i = int(day)
            year_i = today.year
            if (month_i, day_i) > (today.month, today.day):
                year_i -= 1
            return f"{year_i}-{month_i:02d}-{day_i:02d}"
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def parse_html(content: str, base_url: Optional[str] = None) -> List[Dict[str, str]]:
    items = _parse_blocks(content, base_url=base_url)
    if items:
        return items

    parser = _AnchorParser()
    parser.feed(content)
    tokens = parser.tokens
    items = []
    for idx, token in enumerate(tokens):
        if token.get("type") != "a":
            continue
        title = token.get("text", "").strip()
        if not title or title.lower() == "image":
            continue
        if title in _IGNORE_TITLES:
            continue
        href = token.get("href", "").strip()
        url = urljoin(base_url or "", href) if href else ""
        date_text = _normalize_date(title)
        if not date_text:
            for next_token in tokens[idx + 1: idx + 8]:
                if next_token.get("type") != "text":
                    continue
                date_text = _normalize_date(next_token.get("text", ""))
                if date_text:
                    break
        if not date_text:
            for prev_token in tokens[max(0, idx - 8): idx]:
                if prev_token.get("type") != "text":
                    continue
                date_text = _normalize_date(prev_token.get("text", ""))
                if date_text:
                    break
        if not date_text:
            continue
        items.append({
            "title": title,
            "url": url,
            "publish_date": date_text,
            "content": "",
        })
    return items


def _strip_tags(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html_lib.unescape(text)
    return re.sub(r"\\s+", " ", text).strip()


def _extract_href_and_title(block: str) -> Optional[Dict[str, str]]:
    match = re.search(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", block, flags=re.S | re.I)
    if not match:
        return None
    href = match.group(1).strip()
    title = _strip_tags(match.group(2))
    if not title or title.lower() == "image" or title in _IGNORE_TITLES:
        return None
    return {"href": href, "title": title}


def _extract_date_from_block(block: str) -> Optional[str]:
    return _normalize_date(block)


def _parse_blocks(content: str, base_url: Optional[str] = None) -> List[Dict[str, str]]:
    blocks: List[str] = []
    for tag in ["li", "tr", "p", "div"]:
        blocks.extend(re.findall(rf"<{tag}[^>]*>.*?</{tag}>", content, flags=re.S | re.I))
    items: List[Dict[str, str]] = []
    for block in blocks:
        info = _extract_href_and_title(block)
        if not info:
            continue
        date_text = _extract_date_from_block(block)
        if not date_text:
            continue
        url = urljoin(base_url or "", info["href"]) if info["href"] else ""
        items.append({
            "title": info["title"],
            "url": url,
            "publish_date": date_text,
            "content": "",
        })
    return items


def _normalize(df: pd.DataFrame, source: SourceConfig) -> pd.DataFrame:
    if df.empty:
        return df
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df = df.dropna(subset=["publish_date"])
    df["source_name"] = source.name
    df["source_tag"] = source.tag or ""
    df["source_type"] = source.source_type
    return df


def load_sources(config_path: Path) -> List[SourceConfig]:
    payload = load_yaml(config_path)
    raw_sources = payload.get("sources", []) if isinstance(payload, dict) else []
    sources = []
    for item in raw_sources:
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        sources.append(
            SourceConfig(
                name=str(item.get("name") or "未命名"),
                url=url,
                source_type=str(item.get("type") or "rss"),
                tag=item.get("tag") or item.get("source_tag"),
                base_url=item.get("base_url"),
                insecure=bool(item.get("insecure", False)),
            )
        )
    return sources


def build_config(base: Dict, args: argparse.Namespace) -> FetchConfig:
    settings = base.get("settings", {}) if isinstance(base, dict) else {}
    return FetchConfig(
        output_dir=Path("."),
        timeout=int(args.timeout or settings.get("request_timeout", 20)),
        sleep_seconds=float(args.sleep_seconds or settings.get("sleep_seconds", 1.0)),
        user_agent=str(args.user_agent or settings.get("user_agent", "SagePolicyBot/0.1")),
        max_items_per_source=(
            int(args.max_items)
            if args.max_items is not None
            else settings.get("max_items_per_source")
        ),
        dump_html=bool(args.dump_html or settings.get("dump_html", False)),
    )


def fetch_sources(sources: List[SourceConfig], cfg: FetchConfig) -> pd.DataFrame:
    frames = []
    for source in sources:
        if not source.url:
            logger.warning(f"跳过空URL来源: {source.name}")
            continue
        source_type = source.source_type.lower()
        if source_type not in {"rss", "atom", "html"}:
            logger.warning(f"暂不支持的来源类型: {source.source_type} ({source.name})")
            continue
        logger.info(f"拉取 {source.name} ({source.url})")
        content = None
        try:
            content = _fetch_url(source.url, cfg, insecure=source.insecure)
        except Exception as exc:
            logger.warning(f"{source.name} 拉取失败: {exc}")

        if content is None:
            cache_dir = cfg.dump_dir or (cfg.output_dir / "raw_html")
            safe = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", source.tag or source.name)
            for ext in [".html", ".htm", ".xml", ".txt"]:
                cache_path = cache_dir / f"{safe}{ext}"
                if cache_path.exists():
                    content = cache_path.read_text(encoding="utf-8", errors="ignore")
                    logger.info(f"{source.name} 使用缓存HTML: {cache_path}")
                    break
        if content is None:
            logger.warning(f"{source.name} 无可用内容，跳过")
            continue
        if cfg.dump_html and cfg.dump_dir:
            safe = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", source.tag or source.name)
            dump_path = cfg.dump_dir / f"{safe}.html"
            dump_path.write_text(content, encoding="utf-8", errors="ignore")
        if source_type == "html":
            items = parse_html(content, base_url=source.base_url or source.url)
        else:
            items = parse_rss(content)

        if not items:
            cache_dir = cfg.dump_dir or (cfg.output_dir / "raw_html")
            safe = re.sub(r"[^a-zA-Z0-9_\\-]+", "_", source.tag or source.name)
            cache_path = None
            for ext in [".html", ".htm", ".xml", ".txt"]:
                candidate = cache_dir / f"{safe}{ext}"
                if candidate.exists():
                    cache_path = candidate
                    break
            if cache_path:
                cached = cache_path.read_text(encoding="utf-8", errors="ignore")
                logger.info(f"{source.name} 使用缓存重新解析: {cache_path}")
                if source_type == "html":
                    items = parse_html(cached, base_url=source.base_url or source.url)
                else:
                    items = parse_rss(cached)

        if not items:
            logger.warning(f"{source.name} 未解析到有效条目")
            continue
        df = pd.DataFrame(items)
        df = _normalize(df, source)
        if cfg.max_items_per_source:
            df = df.sort_values("publish_date").tail(cfg.max_items_per_source)
        frames.append(df)
        time.sleep(cfg.sleep_seconds)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/policy_sources.yaml", help="来源配置文件")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认 data/raw/policy）")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=None)
    parser.add_argument("--user-agent", type=str, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dump-html", action="store_true", help="保存原始HTML以便排查")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("raw", "policy", ensure=True)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_yaml(config_path)
    sources = load_sources(config_path)
    cfg = build_config(payload, args)
    cfg.output_dir = output_dir
    if cfg.dump_html:
        cfg.dump_dir = output_dir / "raw_html"
        cfg.dump_dir.mkdir(parents=True, exist_ok=True)

    disable_proxy()
    data = fetch_sources(sources, cfg)
    if data.empty:
        logger.warning("未拉取到任何政府政策文本")
        return

    output_path = output_dir / "gov_notices.parquet"
    data.to_parquet(output_path, index=False)
    logger.info(f"政府政策文本已保存: {output_path}")

    summary = {
        "rows": int(len(data)),
        "sources": list(data["source_name"].value_counts().to_dict().items()),
    }
    summary_path = output_dir / "gov_notices_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
