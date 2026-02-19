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
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_data_path, get_data_root, get_tushare_root

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
    # 默认：先读 data/raw/policy，再读 data/tushare/policy（若存在）
    dirs: List[Path] = []
    raw_policy = get_data_path("raw", "policy")
    if raw_policy.exists():
        dirs.append(raw_policy)
    raw_tushare_policy = get_tushare_root() / "policy"
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
    result = pd.DataFrame(
        {
            "publish_date": _extract_date(df),
            "title": _extract_title(df),
            "content": _extract_text(df),
        }
    )
    industry_raw = None
    for col in ["industry", "industry_name", "sw_industry"]:
        if col in df.columns:
            industry_raw = df[col].fillna("").astype(str)
            break
    result["industry_raw"] = industry_raw if industry_raw is not None else ""
    result["source_type"] = source_type
    result = result.dropna(subset=["publish_date"])
    return result


def _normalize_text(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def deduplicate_records(texts: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if texts.empty:
        return texts, {"before": 0, "after": 0, "removed": 0}

    df = texts.copy()
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df = df.dropna(subset=["publish_date"])
    df["trade_date"] = df["publish_date"].dt.floor("D")
    df["title_norm"] = df["title"].astype(str).map(_normalize_text)
    df["content_norm"] = df["content"].astype(str).map(_normalize_text)

    def _doc_id(row: pd.Series) -> str:
        payload = "||".join(
            [
                row["source_type"],
                str(row["trade_date"].date()),
                row["title_norm"],
                row["content_norm"][:1000],
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    df["doc_id"] = df.apply(_doc_id, axis=1)
    before = len(df)
    df = df.drop_duplicates(subset=["doc_id"], keep="first").reset_index(drop=True)
    after = len(df)
    df = df.drop(columns=["title_norm", "content_norm"])
    return df, {"before": before, "after": after, "removed": before - after}


def build_source_health(texts: pd.DataFrame) -> pd.DataFrame:
    if texts.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "source_type",
                "source_doc_count",
                "source_cadence_14d",
                "source_activity_ratio",
                "source_stability_score",
            ]
        )

    frames = []
    for source, group in texts.groupby("source_type"):
        series = (
            group.assign(trade_date=pd.to_datetime(group["publish_date"]).dt.floor("D"))
            .groupby("trade_date")
            .size()
            .rename("source_doc_count")
            .sort_index()
        )
        if series.empty:
            continue
        full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
        series = series.reindex(full_idx, fill_value=0.0)

        cadence = (series > 0).rolling(14, min_periods=1).mean()
        avg_30 = series.rolling(30, min_periods=1).mean()
        activity_ratio = (series / avg_30.replace(0, pd.NA)).fillna(1.0).clip(lower=0.0, upper=2.0)
        stability = (0.5 * cadence + 0.5 * (activity_ratio / 2.0)).clip(lower=0.2, upper=1.0)

        out = pd.DataFrame(
            {
                "trade_date": series.index,
                "source_type": source,
                "source_doc_count": series.values,
                "source_cadence_14d": cadence.values,
                "source_activity_ratio": activity_ratio.values,
                "source_stability_score": stability.values,
            }
        )
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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


def _normalize_industry_name(name: str) -> str:
    return re.sub(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+$", "", name or "").strip()


def load_sw_l2_to_l1_map() -> Dict[str, str]:
    data_root = get_data_root()
    l2_path = data_root / "tushare" / "sectors" / "SW2021_L2_classify.csv"
    l1_path = data_root / "tushare" / "sectors" / "SW2021_L1_classify.csv"
    if not l2_path.exists() or not l1_path.exists():
        return {}
    l2 = pd.read_csv(l2_path)
    l1 = pd.read_csv(l1_path)
    l1_map = dict(zip(l1["industry_code"].astype(str), l1["industry_name"].astype(str)))
    mapping: Dict[str, str] = {}
    for _, row in l2.iterrows():
        name = str(row.get("industry_name", "")).strip()
        parent = str(row.get("parent_code", "")).strip()
        if not name or not parent:
            continue
        l1_name = l1_map.get(parent)
        if l1_name:
            mapping[name] = l1_name
    return mapping


def map_industries(
    raw: str,
    l2_to_l1: Dict[str, str],
    l1_set: set,
    alias_map: Dict[str, str],
) -> List[str]:
    if not raw:
        return []
    parts = re.split(r"[、,;/|]", raw)
    results: List[str] = []
    for part in parts:
        name = part.strip()
        if not name:
            continue
        alias = alias_map.get(name)
        if alias:
            if alias in l1_set:
                results.append(alias)
                continue
            if alias in l2_to_l1:
                results.append(l2_to_l1[alias])
                continue
        if name in l1_set:
            results.append(name)
            continue
        if name in l2_to_l1:
            results.append(l2_to_l1[name])
            continue
        normalized = _normalize_industry_name(name)
        if normalized in l1_set:
            results.append(normalized)
            continue
        if normalized in l2_to_l1:
            results.append(l2_to_l1[normalized])
            continue
    return list(dict.fromkeys(results))


def build_score(text: str, positive: List[str], negative: List[str]) -> Tuple[float, int, int]:
    if not text:
        return 0.5, 0, 0
    pos = sum(1 for kw in positive if kw in text)
    neg = sum(1 for kw in negative if kw in text)
    raw = pos - neg
    score = 0.5 + 0.1 * raw
    return max(0.0, min(1.0, score)), pos, neg


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
    source_health: pd.DataFrame,
    l2_to_l1: Dict[str, str],
    alias_map: Dict[str, str],
) -> pd.DataFrame:
    if texts.empty:
        return pd.DataFrame()

    rows = []
    l1_set = set(l2_to_l1.values()) | set(industry_keywords.keys())
    if not source_health.empty:
        source_health = source_health.copy()
        source_health["trade_date"] = pd.to_datetime(source_health["trade_date"]).dt.floor("D")

    for _, row in texts.iterrows():
        text = f"{row['title']} {row['content']}"
        industries = []
        industry_raw = row.get("industry_raw", "") if isinstance(row, dict) else row.get("industry_raw", "")
        if industry_raw:
            industries = map_industries(str(industry_raw), l2_to_l1, l1_set, alias_map)
        if not industries:
            industries = extract_industries(text, industry_keywords)
        if not industries:
            continue
        base_score, pos_hits, neg_hits = build_score(text, positive, negative)
        source_type = row["source_type"]
        trade_date = pd.to_datetime(row["publish_date"]).floor("D")
        base_weight = float(source_weights.get(source_type, 1.0))
        stability_score = 1.0
        if not source_health.empty:
            matched = source_health[
                (source_health["source_type"] == source_type) & (source_health["trade_date"] == trade_date)
            ]
            if not matched.empty:
                stability_score = float(matched.iloc[-1]["source_stability_score"])
        weight = base_weight * stability_score
        for industry in industries:
            rows.append(
                {
                    "trade_date": trade_date,
                    "sw_industry": industry,
                    "score": base_score,
                    "weight": weight,
                    "base_weight": base_weight,
                    "source_stability_score": stability_score,
                    "source_type": source_type,
                    "pos_hits": pos_hits,
                    "neg_hits": neg_hits,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    def _weighted_mean(group: pd.DataFrame) -> float:
        total = group["weight"].sum()
        if total <= 0:
            return float(group["score"].mean())
        return float((group["score"] * group["weight"]).sum() / total)

    aggregated = df.groupby(["trade_date", "sw_industry"]).apply(_weighted_mean).reset_index(name="policy_score")
    meta = (
        df.groupby(["trade_date", "sw_industry"])
        .agg(
            doc_count=("score", "count"),
            source_count=("source_type", "nunique"),
            source_weights=("weight", "sum"),
            avg_source_stability=("source_stability_score", "mean"),
            score_std=("score", "std"),
            pos_hits=("pos_hits", "sum"),
            neg_hits=("neg_hits", "sum"),
        )
        .reset_index()
    )

    result = aggregated.merge(meta, on=["trade_date", "sw_industry"], how="left")
    result["score_std"] = result["score_std"].fillna(0.0)
    result["sentiment_strength"] = (result["pos_hits"] - result["neg_hits"]).abs()
    doc_component = (result["doc_count"].clip(upper=8) / 8.0) * 0.4
    source_component = (result["source_count"].clip(upper=3) / 3.0) * 0.3
    stability_component = result["avg_source_stability"].clip(lower=0.0, upper=1.0) * 0.2
    dispersion_component = (1.0 - (result["score_std"] / 0.25).clip(lower=0.0, upper=1.0)) * 0.1
    result["confidence"] = (doc_component + source_component + stability_component + dispersion_component).clip(
        0.0, 1.0
    )
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
    texts, dedup_stats = deduplicate_records(texts)
    source_health = build_source_health(texts)

    l2_to_l1 = load_sw_l2_to_l1_map()
    alias_cfg = load_yaml(PROJECT_ROOT / "config" / "policy_industry_alias.yaml")
    alias_map = (alias_cfg.get("aliases") or {}) if isinstance(alias_cfg, dict) else {}
    signals = aggregate_signals(
        texts=texts,
        industry_keywords=industry_keywords,
        positive=positive,
        negative=negative,
        source_weights=source_weights,
        source_health=source_health,
        l2_to_l1=l2_to_l1,
        alias_map=alias_map,
    )
    if signals.empty:
        print("未提取到有效行业政策信号")
        return

    signals.to_parquet(output_dir / "policy_signals.parquet", index=False)
    print(f"政策信号已保存: {output_dir / 'policy_signals.parquet'}")
    if not source_health.empty:
        source_health.to_parquet(output_dir / "policy_source_health.parquet", index=False)
        print(f"来源健康度已保存: {output_dir / 'policy_source_health.parquet'}")

    # 记录来源摘要
    summary = {
        "input_dir": [str(p) for p in input_dirs],
        "rows": int(len(texts)),
        "signals": int(len(signals)),
        "sources": list(texts["source_type"].value_counts().to_dict().items()),
        "dedup": dedup_stats,
    }
    with open(output_dir / "policy_signals_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"摘要已保存: {output_dir / 'policy_signals_summary.json'}")


if __name__ == "__main__":
    main()
