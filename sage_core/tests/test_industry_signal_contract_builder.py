from __future__ import annotations

import pandas as pd

from scripts.strategy.build_industry_signal_contract import (
    _build_aligned_signal_snapshot,
    _to_snapshot,
    build_industry_signal_contract_artifacts,
)


def _sample_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": "2026-02-14",
                "sw_industry": "电子",
                "signal_name": "policy_score",
                "score": 0.8,
                "confidence": 0.9,
                "direction": 1,
                "source": "policy",
                "model_version": "v1",
                "meta": "{}",
            },
            {
                "trade_date": "2026-02-14",
                "sw_industry": "煤炭",
                "signal_name": "policy_score",
                "score": 0.2,
                "confidence": 0.8,
                "direction": -1,
                "source": "policy",
                "model_version": "v1",
                "meta": "{}",
            },
            {
                "trade_date": "2026-02-13",
                "sw_industry": "电子",
                "signal_name": "concept_bias",
                "score": 0.75,
                "confidence": 0.7,
                "direction": 1,
                "source": "concept",
                "model_version": "v1",
                "meta": "{}",
            },
            {
                "trade_date": "2026-01-02",
                "sw_industry": "电子",
                "signal_name": "northbound_ratio",
                "score": 0.65,
                "confidence": 0.6,
                "direction": 1,
                "source": "northbound",
                "model_version": "v1",
                "meta": "{}",
            },
        ]
    )


def test_build_aligned_signal_snapshot_with_lookback():
    contract = _sample_contract()
    aligned = _build_aligned_signal_snapshot(
        contract=contract,
        as_of_date=pd.Timestamp("2026-02-14"),
        lookback_days={"policy_score": 3, "concept_bias": 7, "northbound_ratio": 45},
        default_lookback_days=30,
        confidence_half_life_days={"policy_score": 5, "concept_bias": 5, "northbound_ratio": 7},
        default_confidence_half_life_days=7.0,
    )
    assert not aligned.empty
    assert aligned["trade_date"].nunique() == 1
    assert aligned["trade_date"].iloc[0] == pd.Timestamp("2026-02-14")
    assert set(aligned["signal_name"]) == {"policy_score", "concept_bias", "northbound_ratio"}

    concept_row = aligned[aligned["signal_name"] == "concept_bias"].iloc[0]
    northbound_row = aligned[aligned["signal_name"] == "northbound_ratio"].iloc[0]
    assert concept_row["stale_days"] == 1
    assert northbound_row["stale_days"] == 43
    assert northbound_row["source_trade_date"] == pd.Timestamp("2026-01-02")
    assert northbound_row["confidence"] < northbound_row["confidence_raw"]


def test_build_aligned_signal_snapshot_drops_expired_signal():
    contract = _sample_contract()
    aligned = _build_aligned_signal_snapshot(
        contract=contract,
        as_of_date=pd.Timestamp("2026-02-14"),
        lookback_days={"policy_score": 3, "concept_bias": 7, "northbound_ratio": 30},
        default_lookback_days=30,
        confidence_half_life_days={"policy_score": 5, "concept_bias": 5, "northbound_ratio": 7},
        default_confidence_half_life_days=7.0,
    )
    assert "northbound_ratio" not in set(aligned["signal_name"])


def test_to_snapshot_generates_rank():
    contract = _sample_contract()
    aligned = _build_aligned_signal_snapshot(
        contract=contract,
        as_of_date=pd.Timestamp("2026-02-14"),
        lookback_days={"policy_score": 3, "concept_bias": 7, "northbound_ratio": 45},
        default_lookback_days=30,
        confidence_half_life_days={"policy_score": 5, "concept_bias": 5, "northbound_ratio": 7},
        default_confidence_half_life_days=7.0,
    )
    score_snapshot = _to_snapshot(aligned)
    assert {"trade_date", "sw_industry", "combined_score", "rank"}.issubset(score_snapshot.columns)
    assert score_snapshot["trade_date"].nunique() == 1


def test_build_industry_signal_contract_artifacts(tmp_path):
    policy = pd.DataFrame(
        [
            {"trade_date": "2026-02-14", "sw_industry": "电子", "policy_score": 0.8, "doc_count": 2, "source_count": 1},
            {"trade_date": "2026-02-14", "sw_industry": "煤炭", "policy_score": 0.3, "doc_count": 2, "source_count": 1},
        ]
    )
    concept = pd.DataFrame(
        [
            {
                "trade_date": "2026-02-13",
                "sw_industry": "电子",
                "concept_bias_strength": 0.4,
                "concept_signal_confidence": 0.8,
                "concept_count": 1,
                "mean_heat_score": 1.2,
                "overheat_rate": 0.0,
            }
        ]
    )
    northbound = pd.DataFrame([{"trade_date": "2026-01-20", "industry_name": "电子", "ratio_signal": 0.2}])
    policy_path = tmp_path / "policy.parquet"
    concept_path = tmp_path / "concept.parquet"
    northbound_path = tmp_path / "northbound.parquet"
    output_dir = tmp_path / "signals"
    policy.to_parquet(policy_path, index=False)
    concept.to_parquet(concept_path, index=False)
    northbound.to_parquet(northbound_path, index=False)

    result = build_industry_signal_contract_artifacts(
        output_dir=output_dir,
        policy_path=policy_path,
        concept_bias_path=concept_path,
        northbound_path=northbound_path,
        as_of_date="2026-02-14",
        signal_lookback_days={"policy_score": 3, "concept_bias": 7, "northbound_ratio": 30},
        signal_half_life_days={"policy_score": 5, "concept_bias": 5, "northbound_ratio": 7},
    )
    assert result["generated"] is True
    snapshot = pd.read_parquet(result["signal_snapshot_path"])
    assert set(snapshot["signal_name"]) == {"policy_score", "concept_bias", "northbound_ratio"}
    assert set(snapshot["trade_date"].dt.strftime("%Y-%m-%d")) == {"2026-02-14"}
