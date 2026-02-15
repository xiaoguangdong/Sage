#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_data_path

DEFAULT_SIGNAL_LOOKBACK_DAYS = {
    "policy_score": 3,
    "concept_bias": 7,
    "northbound_ratio": 45,
}
DEFAULT_SIGNAL_HALF_LIFE_DAYS = {
    "policy_score": 5,
    "concept_bias": 5,
    "northbound_ratio": 7,
}


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


def _parse_signal_lookback(raw: str | None) -> Dict[str, int]:
    if not raw:
        return {}
    mapping: Dict[str, int] = {}
    for segment in raw.split(","):
        item = segment.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"signal lookback 配置非法: {item}")
        signal_name, days = item.split("=", 1)
        signal_name = signal_name.strip()
        if not signal_name:
            raise ValueError(f"signal lookback 缺少 signal_name: {item}")
        mapping[signal_name] = int(days.strip())
    return mapping


def _parse_signal_float_map(raw: str | None) -> Dict[str, float]:
    if not raw:
        return {}
    mapping: Dict[str, float] = {}
    for segment in raw.split(","):
        item = segment.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"signal float map 配置非法: {item}")
        signal_name, value = item.split("=", 1)
        signal_name = signal_name.strip()
        if not signal_name:
            raise ValueError(f"signal float map 缺少 signal_name: {item}")
        mapping[signal_name] = float(value.strip())
    return mapping


def _build_aligned_signal_snapshot(
    contract: pd.DataFrame,
    as_of_date: pd.Timestamp,
    lookback_days: Dict[str, int],
    default_lookback_days: int,
    confidence_half_life_days: Dict[str, float],
    default_confidence_half_life_days: float,
) -> pd.DataFrame:
    if contract.empty:
        return pd.DataFrame()

    as_of_date = pd.Timestamp(as_of_date).floor("D")
    source = contract.copy()
    source["trade_date"] = pd.to_datetime(source["trade_date"]).dt.floor("D")
    source = source[source["trade_date"] <= as_of_date].copy()
    if source.empty:
        return pd.DataFrame()

    source["max_stale_days"] = (
        source["signal_name"].map(lookback_days).fillna(int(default_lookback_days)).astype(int)
    )
    source["stale_days"] = (as_of_date - source["trade_date"]).dt.days
    source = source[(source["stale_days"] >= 0) & (source["stale_days"] <= source["max_stale_days"])].copy()
    if source.empty:
        return pd.DataFrame()

    source = source.sort_values(["signal_name", "sw_industry", "trade_date"])
    latest = source.groupby(["signal_name", "sw_industry"], as_index=False).tail(1).copy()
    latest["confidence_raw"] = pd.to_numeric(latest["confidence"], errors="coerce").fillna(0.0)
    latest["confidence_half_life_days"] = (
        latest["signal_name"]
        .map(confidence_half_life_days)
        .fillna(float(default_confidence_half_life_days))
        .clip(lower=1.0)
    )
    latest["confidence_decay"] = np.exp(-np.log(2.0) * latest["stale_days"] / latest["confidence_half_life_days"])
    latest["confidence"] = (latest["confidence_raw"] * latest["confidence_decay"]).clip(0.0, 1.0)
    latest["source_trade_date"] = latest["trade_date"]
    latest["trade_date"] = as_of_date
    return latest[
        [
            "trade_date",
            "source_trade_date",
            "stale_days",
            "max_stale_days",
            "confidence_raw",
            "confidence_half_life_days",
            "confidence_decay",
            "sw_industry",
            "signal_name",
            "score",
            "confidence",
            "direction",
            "source",
            "model_version",
            "meta",
        ]
    ].sort_values(["signal_name", "sw_industry"]).reset_index(drop=True)


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


def build_industry_signal_contract_artifacts(
    *,
    output_dir: Path,
    policy_path: Path | None = None,
    concept_bias_path: Path | None = None,
    northbound_path: Path | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    signal_lookback_days: Dict[str, int] | None = None,
    default_lookback_days: int = 30,
    signal_half_life_days: Dict[str, float] | None = None,
    default_half_life_days: float = 7.0,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_default = get_data_path("processed", "policy", "policy_signals_enhanced.parquet")
    if not policy_default.exists():
        policy_default = get_data_path("processed", "policy", "policy_signals.parquet")
    policy_path = Path(policy_path) if policy_path else policy_default

    concept_bias_default = get_data_path("signals", "industry", "industry_concept_bias.parquet")
    concept_bias_path = Path(concept_bias_path) if concept_bias_path else concept_bias_default

    northbound_default = _latest_by_prefix(get_data_path("signals"), "northbound_industry_ratio_")
    northbound_path = Path(northbound_path) if northbound_path else northbound_default

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
        return {
            "generated": False,
            "reason": "no_signal_frames",
            "output_dir": str(output_dir),
            "contract_path": str(output_dir / "industry_signal_contract.parquet"),
            "signal_snapshot_path": str(output_dir / "industry_signal_snapshot_latest.parquet"),
            "score_snapshot_path": str(output_dir / "industry_score_snapshot_latest.parquet"),
            "summary_path": str(output_dir / "industry_signal_contract_summary.json"),
            "summary": {},
        }

    contract = _normalize_contract(pd.concat(frames, ignore_index=True))
    contract_path = output_dir / "industry_signal_contract.parquet"
    contract.to_parquet(contract_path, index=False)

    lookback_days = dict(DEFAULT_SIGNAL_LOOKBACK_DAYS)
    if signal_lookback_days:
        lookback_days.update({str(k): int(v) for k, v in signal_lookback_days.items()})
    half_life_days = dict(DEFAULT_SIGNAL_HALF_LIFE_DAYS)
    if signal_half_life_days:
        half_life_days.update({str(k): float(v) for k, v in signal_half_life_days.items()})

    if as_of_date is not None:
        as_of = pd.to_datetime(as_of_date).floor("D")
    else:
        as_of = contract["trade_date"].max()

    signal_snapshot = _build_aligned_signal_snapshot(
        contract=contract,
        as_of_date=as_of,
        lookback_days=lookback_days,
        default_lookback_days=default_lookback_days,
        confidence_half_life_days=half_life_days,
        default_confidence_half_life_days=default_half_life_days,
    )
    score_snapshot = _to_snapshot(signal_snapshot) if not signal_snapshot.empty else pd.DataFrame()

    signal_snapshot_path = output_dir / "industry_signal_snapshot_latest.parquet"
    signal_snapshot_date_path = output_dir / f"industry_signal_snapshot_{as_of.strftime('%Y%m%d')}.parquet"
    if not signal_snapshot.empty:
        signal_snapshot.to_parquet(signal_snapshot_path, index=False)
        signal_snapshot.to_parquet(signal_snapshot_date_path, index=False)

    score_snapshot_path = output_dir / "industry_score_snapshot_latest.parquet"
    score_snapshot_date_path = output_dir / f"industry_score_snapshot_{as_of.strftime('%Y%m%d')}.parquet"
    if not score_snapshot.empty:
        score_snapshot.to_parquet(score_snapshot_path, index=False)
        score_snapshot.to_parquet(score_snapshot_date_path, index=False)

    signal_freshness: Dict[str, Dict[str, float | int | None]] = {}
    if not signal_snapshot.empty:
        for signal_name, group in signal_snapshot.groupby("signal_name"):
            signal_freshness[str(signal_name)] = {
                "rows": int(len(group)),
                "max_stale_days": int(group["stale_days"].max()),
                "avg_stale_days": round(float(group["stale_days"].mean()), 4),
                "avg_confidence_raw": round(float(group["confidence_raw"].mean()), 4),
                "avg_confidence_effective": round(float(group["confidence"].mean()), 4),
            }

    summary = {
        "rows_contract": int(len(contract)),
        "rows_signal_snapshot": int(len(signal_snapshot)),
        "rows_score_snapshot": int(len(score_snapshot)),
        "signal_names": sorted(contract["signal_name"].unique().tolist()),
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "latest_date": None if signal_snapshot.empty else signal_snapshot["trade_date"].max().strftime("%Y-%m-%d"),
        "sources": sorted(contract["source"].unique().tolist()),
        "signal_lookback_days": lookback_days,
        "default_lookback_days": int(default_lookback_days),
        "signal_half_life_days": half_life_days,
        "default_half_life_days": float(default_half_life_days),
        "signal_freshness": signal_freshness,
    }

    summary_path = output_dir / "industry_signal_contract_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "generated": True,
        "output_dir": str(output_dir),
        "contract_path": str(contract_path),
        "signal_snapshot_path": str(signal_snapshot_path),
        "signal_snapshot_date_path": str(signal_snapshot_date_path),
        "score_snapshot_path": str(score_snapshot_path),
        "score_snapshot_date_path": str(score_snapshot_date_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="行业信号统一契约生成")
    parser.add_argument("--policy-path", type=str, default=None)
    parser.add_argument("--concept-bias-path", type=str, default=None)
    parser.add_argument("--northbound-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--as-of-date", type=str, default=None, help="决策日，格式 YYYYMMDD 或 YYYY-MM-DD，默认使用契约最大日期")
    parser.add_argument(
        "--signal-lookback-days",
        type=str,
        default=None,
        help="按信号配置最大回看天数，例如 policy_score=3,concept_bias=7,northbound_ratio=45",
    )
    parser.add_argument("--default-lookback-days", type=int, default=30, help="未单独配置信号的默认最大回看天数")
    parser.add_argument(
        "--signal-half-life-days",
        type=str,
        default=None,
        help="按信号配置置信度半衰期（天），例如 policy_score=5,concept_bias=5,northbound_ratio=7",
    )
    parser.add_argument("--default-half-life-days", type=float, default=7.0, help="未单独配置信号的置信度半衰期（天）")
    args = parser.parse_args()
    lookback_days = _parse_signal_lookback(args.signal_lookback_days)
    half_life_days = _parse_signal_float_map(args.signal_half_life_days)
    output_dir = Path(args.output_dir) if args.output_dir else get_data_path("signals", "industry", ensure=True)
    result = build_industry_signal_contract_artifacts(
        output_dir=output_dir,
        policy_path=Path(args.policy_path) if args.policy_path else None,
        concept_bias_path=Path(args.concept_bias_path) if args.concept_bias_path else None,
        northbound_path=Path(args.northbound_path) if args.northbound_path else None,
        as_of_date=args.as_of_date,
        signal_lookback_days=lookback_days,
        default_lookback_days=args.default_lookback_days,
        signal_half_life_days=half_life_days,
        default_half_life_days=args.default_half_life_days,
    )
    if not result.get("generated", False):
        print("无可用行业信号，未生成契约")
        return
    print(f"已保存: {result['contract_path']}")
    print(f"摘要: {result['summary_path']}")


if __name__ == "__main__":
    main()
