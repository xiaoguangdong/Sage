#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政策信号管道（MVP）

功能：
1) 读取政策文本（Tushare公告 / 政府网站 / 研报）
2) 规则抽取行业 + 情绪分数（0-1）
3) 输出行业级日频政策信号

说明：
- 仅做规则版，后续可替换为LLM/NLP
- 输出字段：trade_date, sw_industry, policy_score, doc_count, source_weights
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_data_path


DEFAULT_INPUT_NAMES = {
    "tushare_announcement": [
        "tushare_anns.parquet",
        "tushare_anns.csv",
        "tushare_announcements.parquet",
        "tushare_announcements.csv",
    ],
    "gov_notice": [
        "gov_notices.parquet",
        "gov_notices.csv",
        "ndrc_notices.parquet",
        "miit_notices.parquet",
        "state_council_notices.parquet",
    ],
    "research_report": [
        "tushare_reports.parquet",
        "tushare_reports.csv",
        "tushare_report.parquet",
        "tushare_report.csv",
        "eastmoney_industry_reports.parquet",
        "eastmoney_industry_reports.csv",
    ],
}


def load_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def resolve_input_dirs(input_dir: Optional[str]) -> List[Path]:
    if input_dir:
        path = Path(input_dir)
        return [path if path.is_absolute() else PROJECT_ROOT / path]
    # 默认：先读 data/raw/policy，再读 data/raw/tushare/policy（若存在）
    dirs: List[Path] = []
    raw_policy = get_data_path("raw", "policy")
    if raw_policy.exists():
        dirs.append(raw_policy)
    raw_tushare_policy = get_data_path("raw", "tushare", "policy")
    if raw_tushare_policy.exists():
        dirs.append(raw_tushare_policy)
    return dirs or [raw_policy]


def detect_source_files(input_dir: Path) -> Dict[str, Path]:
    files = {}
    for source, names in DEFAULT_INPUT_NAMES.items():
        for name in names:
            path = input_dir / name
            if path.exists():
                files[source] = path
                break
    return files


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"不支持的文件格式: {path}")


def _extract_date(df: pd.DataFrame) -> pd.Series:
    for col in ["publish_date", "pub_date", "ann_date", "date", "trade_date", "created_date"]:
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")
    return pd.to_datetime(pd.Series([None] * len(df)))


def _extract_text(df: pd.DataFrame) -> pd.Series:
    for col in ["content", "summary", "text", "body", "description"]:
        if col in df.columns:
            return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df))


def _extract_title(df: pd.DataFrame) -> pd.Series:
    for col in ["title", "headline", "subject"]:
        if col in df.columns:
            return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df))


def normalize_records(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    result = pd.DataFrame({
        "publish_date": _extract_date(df),
        "title": _extract_title(df),
        "content": _extract_text(df),
    })
    result["source_type"] = source_type
    result = result.dropna(subset=["publish_date"])
    return result


def load_policy_texts(input_dirs: List[Path]) -> pd.DataFrame:
    frames = []
    for input_dir in input_dirs:
        files = detect_source_files(input_dir)
        if not files:
            continue
        for source, path in files.items():
            df = _read_any(path)
            frames.append(normalize_records(df, source))
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["publish_date"] = pd.to_datetime(data["publish_date"])
    data = data.dropna(subset=["publish_date"])
    return data


def build_score(text: str, positive: List[str], negative: List[str]) -> float:
    if not text:
        return 0.5
    pos = sum(1 for kw in positive if kw in text)
    neg = sum(1 for kw in negative if kw in text)
    raw = pos - neg
    score = 0.5 + 0.1 * raw
    return max(0.0, min(1.0, score))


def extract_industries(text: str, mapping: Dict[str, List[str]]) -> List[str]:
    if not text:
        return []
    found = []
    for industry, keywords in mapping.items():
        if any(k in text for k in keywords):
            found.append(industry)
    return found


def aggregate_signals(
    texts: pd.DataFrame,
    industry_keywords: Dict[str, List[str]],
    positive: List[str],
    negative: List[str],
    source_weights: Dict[str, float],
) -> pd.DataFrame:
    if texts.empty:
        return pd.DataFrame()

    rows = []
    for _, row in texts.iterrows():
        text = f"{row['title']} {row['content']}"
        industries = extract_industries(text, industry_keywords)
        if not industries:
            continue
        base_score = build_score(text, positive, negative)
        weight = float(source_weights.get(row["source_type"], 1.0))
        for industry in industries:
            rows.append({
                "trade_date": row["publish_date"].date(),
                "sw_industry": industry,
                "score": base_score,
                "weight": weight,
                "source_type": row["source_type"],
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    def _weighted_mean(group: pd.DataFrame) -> float:
        total = group["weight"].sum()
        if total <= 0:
            return float(group["score"].mean())
        return float((group["score"] * group["weight"]).sum() / total)

    aggregated = (
        df.groupby(["trade_date", "sw_industry"])
        .apply(_weighted_mean)
        .reset_index(name="policy_score")
    )
    meta = df.groupby(["trade_date", "sw_industry"]).agg(
        doc_count=("score", "count"),
        source_weights=("weight", "sum"),
    ).reset_index()

    result = aggregated.merge(meta, on=["trade_date", "sw_industry"], how="left")
    return result.sort_values(["trade_date", "policy_score"], ascending=[True, False])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=None, help="政策文本输入目录")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录（默认 data/processed/policy）")
    args = parser.parse_args()

    input_dirs = resolve_input_dirs(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("processed", "policy", ensure=True)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    keyword_cfg = load_yaml(PROJECT_ROOT / "config" / "policy_keywords.yaml")
    industry_cfg = load_yaml(PROJECT_ROOT / "config" / "policy_industry_keywords.yaml")
    positive = (keyword_cfg.get("sentiment") or {}).get("positive", [])
    negative = (keyword_cfg.get("sentiment") or {}).get("negative", [])
    source_weights = keyword_cfg.get("source_weights", {})
    industry_keywords = industry_cfg.get("industry_keywords", {})

    texts = load_policy_texts(input_dirs)
    if texts.empty:
        print(f"未发现政策文本文件，目录: {', '.join(str(p) for p in input_dirs)}")
        return

    signals = aggregate_signals(texts, industry_keywords, positive, negative, source_weights)
    if signals.empty:
        print("未提取到有效行业政策信号")
        return

    signals.to_parquet(output_dir / "policy_signals.parquet", index=False)
    print(f"政策信号已保存: {output_dir / 'policy_signals.parquet'}")

    # 记录来源摘要
    summary = {
        "input_dir": [str(p) for p in input_dirs],
        "rows": int(len(texts)),
        "signals": int(len(signals)),
        "sources": list(texts["source_type"].value_counts().to_dict().items()),
    }
    with open(output_dir / "policy_signals_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"摘要已保存: {output_dir / 'policy_signals_summary.json'}")


if __name__ == "__main__":
    main()
