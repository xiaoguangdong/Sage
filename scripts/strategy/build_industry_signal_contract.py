#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_data_path


def _latest_by_prefix(directory: Path, prefix: str) -> Path | None:
    files = sorted(directory.glob(f"{prefix}*.parquet"))
    if not files:
        return None
    return files[-1]


def _load_policy_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.floor("D")
    if "confidence" not in df.columns:
        doc = df["doc_count"].fillna(0.0) if "doc_count" in df.columns else pd.Series(0.0, index=df.index)
        src = df["source_count"].fillna(1.0) if "source_count" in df.columns else pd.Series(1.0, index=df.index)
        df["confidence"] = ((doc.clip(upper=8) / 8.0) * 0.7 + (src.clip(upper=3) / 3.0) * 0.3).clip(0.0, 1.0)
    df["score"] = df["policy_score"].clip(0.0, 1.0)
    df["direction"] = 0
    df.loc[df["score"] > 0.55, "direction"] = 1
    df.loc[df["score"] < 0.45, "direction"] = -1
    meta_cols = [c for c in ["doc_count", "source_count", "sentiment_strength"] if c in df.columns]
    if meta_cols:
        df["meta"] = df[meta_cols].to_dict(orient="records")
    else:
        df["meta"] = [{} for _ in range(len(df))]
    return pd.DataFrame(
        {
            "trade_date": df["trade_date"],
            "sw_industry": df["sw_industry"],
            "signal_name": "policy_score",
            "score": df["score"],
            "confidence": df["confidence"].clip(0.0, 1.0),
            "direction": df["direction"],
            "source": "policy_signal_pipeline",
            "model_version": "policy_v1",
            "meta": df["meta"],
        }
    )


def _load_concept_bias(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.floor("D")
    bias = df["concept_bias_strength"].clip(-1.0, 1.0)
    score = ((bias + 1.0) / 2.0).clip(0.0, 1.0)
    confidence = df["concept_signal_confidence"].clip(0.0, 1.0)
    direction = bias.apply(lambda x: 1 if x > 0.05 else (-1 if x < -0.05 else 0))
    meta = df[["concept_count", "mean_heat_score", "overheat_rate"]].to_dict(orient="records")
    return pd.DataFrame(
        {
            "trade_date": df["trade_date"],
            "sw_industry": df["sw_industry"],
            "signal_name": "concept_bias",
            "score": score,
            "confidence": confidence,
            "direction": direction,
            "source": "ths_concept_bias",
            "model_version": "concept_v1",
            "meta": meta,
        }
    )


def _load_northbound_signal(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.floor("D")
    if "ratio_signal" in df.columns:
        score = ((pd.to_numeric(df["ratio_signal"], errors="coerce").fillna(0.0) + 1.0) / 2.0).clip(0.0, 1.0)
        direction = pd.to_numeric(df["ratio_signal"], errors="coerce").fillna(0.0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    else:
        ratio = pd.to_numeric(df.get("industry_ratio", 0.0), errors="coerce").fillna(0.0)
        score = ratio.rank(pct=True).clip(0.0, 1.0)
        direction = score.apply(lambda x: 1 if x > 0.6 else (-1 if x < 0.4 else 0))
    confidence = pd.Series(0.6, index=df.index)
    meta = df[[c for c in ["industry_ratio", "ratio_signal"] if c in df.columns]].to_dict(orient="records")
    return pd.DataFrame(
        {
            "trade_date": df["trade_date"],
            "sw_industry": df["industry_name"],
            "signal_name": "northbound_ratio",
            "score": score,
            "confidence": confidence,
            "direction": direction,
            "source": "macro_northbound",
            "model_version": "northbound_v1",
            "meta": meta,
        }
    )


def _to_snapshot(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return pd.DataFrame()

    def _weighted_score(group: pd.DataFrame) -> float:
        total = group["confidence"].sum()
        if total <= 0:
            return float(group["score"].mean())
        return float((group["score"] * group["confidence"]).sum() / total)

    snapshot = (
        contract.groupby(["trade_date", "sw_industry"])
        .apply(_weighted_score)
        .reset_index(name="combined_score")
    )
    snapshot["rank"] = snapshot.groupby("trade_date")["combined_score"].rank(ascending=False, method="first")
    return snapshot.sort_values(["trade_date", "combined_score"], ascending=[True, False]).reset_index(drop=True)


def _normalize_contract(contract: pd.DataFrame) -> pd.DataFrame:
    if contract.empty:
        return contract
    contract = contract.copy()
    contract["trade_date"] = pd.to_datetime(contract["trade_date"]).dt.floor("D")
    contract["sw_industry"] = contract["sw_industry"].astype(str)
    contract["signal_name"] = contract["signal_name"].astype(str)
    contract["score"] = pd.to_numeric(contract["score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    contract["confidence"] = pd.to_numeric(contract["confidence"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    contract["direction"] = pd.to_numeric(contract["direction"], errors="coerce").fillna(0).astype(int)
    contract["source"] = contract["source"].astype(str)
    contract["model_version"] = contract["model_version"].astype(str)
    contract["meta"] = contract["meta"].apply(lambda x: json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x)
    return contract[
        [
            "trade_date",
            "sw_industry",
            "signal_name",
            "score",
            "confidence",
            "direction",
            "source",
            "model_version",
            "meta",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="行业信号统一契约生成")
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--concept-bias-path", type=str, default=None)
    parser.add_argument("--northbound-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("signals", "industry", ensure=True)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_default = get_data_path("processed", "policy", "policy_signals_enhanced.parquet")
    if not policy_default.exists():
        policy_default = get_data_path("processed", "policy", "policy_signals.parquet")
    policy_path = Path(args.policy_path) if args.policy_path else policy_default

    concept_bias_default = get_data_path("signals", "industry", "industry_concept_bias.parquet")
    concept_bias_path = Path(args.concept_bias_path) if args.concept_bias_path else concept_bias_default

    northbound_default = _latest_by_prefix(get_data_path("signals"), "northbound_industry_ratio_")
    northbound_path = Path(args.northbound_path) if args.northbound_path else northbound_default

    frames: List[pd.DataFrame] = []
    policy = _load_policy_signals(policy_path)
    if not policy.empty:
        frames.append(policy)
    concept = _load_concept_bias(concept_bias_path)
    if not concept.empty:
        frames.append(concept)
    northbound = _load_northbound_signal(northbound_path) if northbound_path else pd.DataFrame()
    if not northbound.empty:
        frames.append(northbound)

    if not frames:
        print("无可用行业信号，未生成契约")
        return

    contract = _normalize_contract(pd.concat(frames, ignore_index=True))
    contract_path = output_dir / "industry_signal_contract.parquet"
    contract.to_parquet(contract_path, index=False)

    snapshot = _to_snapshot(contract)
    latest_snapshot = pd.DataFrame()
    if not snapshot.empty:
        latest_date = snapshot["trade_date"].max()
        latest_snapshot = snapshot[snapshot["trade_date"] == latest_date].copy()
        latest_snapshot.to_parquet(output_dir / "industry_signal_snapshot_latest.parquet", index=False)
        latest_snapshot.to_parquet(output_dir / f"industry_signal_snapshot_{latest_date.strftime('%Y%m%d')}.parquet", index=False)

    summary = {
        "rows_contract": int(len(contract)),
        "rows_snapshot": int(len(snapshot)),
        "signal_names": sorted(contract["signal_name"].unique().tolist()),
        "latest_date": None if latest_snapshot.empty else latest_snapshot["trade_date"].max().strftime("%Y-%m-%d"),
        "sources": sorted(contract["source"].unique().tolist()),
    }
    (output_dir / "industry_signal_contract_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已保存: {contract_path}")
    print(f"摘要: {output_dir / 'industry_signal_contract_summary.json'}")


if __name__ == "__main__":
    main()
