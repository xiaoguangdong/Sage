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
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from scripts.data._shared.runtime import disable_proxy, get_data_path, setup_logger


logger = setup_logger(Path(__file__).stem, module="data")


@dataclass
class SourceConfig:
    name: str
    url: str
    source_type: str = "rss"
    tag: Optional[str] = None


@dataclass
class FetchConfig:
    output_dir: Path
    timeout: int = 20
    sleep_seconds: float = 1.0
    user_agent: str = "SagePolicyBot/0.1"
    max_items_per_source: Optional[int] = None


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


def _fetch_url(url: str, cfg: FetchConfig) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": cfg.user_agent})
    with urllib.request.urlopen(req, timeout=cfg.timeout) as resp:
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


def _normalize(df: pd.DataFrame, source: SourceConfig) -> pd.DataFrame:
    if df.empty:
        return df
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df = df.dropna(subset=["publish_date"])
    df["source_name"] = source.name
    df["source_tag"] = source.tag or ""
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
    )


def fetch_sources(sources: List[SourceConfig], cfg: FetchConfig) -> pd.DataFrame:
    frames = []
    for source in sources:
        if not source.url:
            logger.warning(f"跳过空URL来源: {source.name}")
            continue
        if source.source_type.lower() not in {"rss", "atom"}:
            logger.warning(f"暂不支持的来源类型: {source.source_type} ({source.name})")
            continue
        logger.info(f"拉取 {source.name} ({source.url})")
        content = _fetch_url(source.url, cfg)
        items = parse_rss(content)
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
