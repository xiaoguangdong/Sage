from __future__ import annotations

from typing import Any

DEFAULT_SIGNAL_WEIGHTS = {
    "policy_score": 0.4,
    "concept_bias": 0.3,
    "northbound_ratio": 0.3,
}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalize_weights(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return dict(DEFAULT_SIGNAL_WEIGHTS)
    weights: dict[str, float] = {}
    for key, raw in value.items():
        try:
            weight = float(raw)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        weights[str(key)] = weight
    return weights or dict(DEFAULT_SIGNAL_WEIGHTS)


def _normalize_industry_tilts(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    tilts: dict[str, float] = {}
    for key, raw in value.items():
        try:
            tilt = float(raw)
        except (TypeError, ValueError):
            continue
        if tilt == 0:
            continue
        tilts[str(key)] = max(-1.0, min(1.0, tilt))
    return tilts


def trend_state_to_regime(trend_state: int) -> str:
    return {0: "bear", 1: "sideways", 2: "bull"}.get(int(trend_state), "sideways")


def resolve_industry_overlay_config(industry_cfg: dict | None, trend_state: int) -> dict[str, Any]:
    cfg = industry_cfg if isinstance(industry_cfg, dict) else {}
    overlay_cfg = cfg.get("overlay") if isinstance(cfg.get("overlay"), dict) else {}
    regime_name = trend_state_to_regime(trend_state)

    regime_overrides = overlay_cfg.get("regime_overrides")
    if not isinstance(regime_overrides, dict):
        regime_overrides = {}
    regime_cfg = regime_overrides.get(regime_name)
    if not isinstance(regime_cfg, dict):
        regime_cfg = {}

    overlay_strength = _safe_float(regime_cfg.get("overlay_strength", overlay_cfg.get("overlay_strength", 0.20)), 0.20)
    overlay_strength = max(0.0, overlay_strength)

    mainline_strength = _safe_float(
        regime_cfg.get("mainline_strength", overlay_cfg.get("mainline_strength", 0.35)), 0.35
    )
    mainline_strength = min(1.0, max(0.0, mainline_strength))

    base_weights = overlay_cfg.get("signal_weights")
    weights_raw = regime_cfg.get("signal_weights", base_weights)
    signal_weights = _normalize_weights(weights_raw)

    tilt_strength = _safe_float(
        regime_cfg.get("tilt_strength", overlay_cfg.get("tilt_strength", 0.0)),
        0.0,
    )
    tilt_strength = min(1.0, max(0.0, tilt_strength))

    base_tilts = _normalize_industry_tilts(overlay_cfg.get("industry_tilts"))
    regime_tilts = _normalize_industry_tilts(regime_cfg.get("industry_tilts"))
    industry_tilts = {**base_tilts, **regime_tilts}

    return {
        "regime_name": regime_name,
        "overlay_strength": overlay_strength,
        "mainline_strength": mainline_strength,
        "signal_weights": signal_weights,
        "tilt_strength": tilt_strength,
        "industry_tilts": industry_tilts,
    }
