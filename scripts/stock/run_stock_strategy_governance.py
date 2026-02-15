#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data._shared.runtime import get_data_path, get_tushare_root
from sage_core.models.stock_selector import SelectionConfig
from sage_core.models.strategy_governance import (
    ChampionChallengerEngine,
    ChallengerConfig,
    MultiAlphaChallengerStrategies,
    SeedBalanceStrategy,
    StrategyGovernanceConfig,
    normalize_strategy_id,
    save_strategy_outputs,
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _load_seed_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"seed输入文件不存在: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"不支持的seed输入格式: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行Champion/Challenger四策略选股")
    parser.add_argument("--trade-date", required=True, help="交易日 YYYYMMDD")
    parser.add_argument("--top-n", type=int, default=10, help="每个策略输出TopN")
    parser.add_argument("--config", type=str, default="sage_app/config/strategy_governance.yaml", help="治理配置路径")
    parser.add_argument("--seed-input", type=str, default=None, help="seed策略输入（parquet/csv）")
    parser.add_argument("--data-dir", type=str, default=None, help="Tushare原始数据根目录（给challenger用）")
    parser.add_argument("--active-champion-id", type=str, default=None, help="手动指定冠军策略ID")
    parser.add_argument("--allocation-method", type=str, default="fixed", choices=["fixed", "regime"])
    parser.add_argument("--regime", type=str, default="sideways", choices=["bear", "sideways", "bull"])
    parser.add_argument("--output-root", type=str, default=None, help="输出目录，默认 data/signals/stock_selector")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = _load_yaml(config_path)

    governance_raw = cfg.get("strategy_governance", {})
    governance = StrategyGovernanceConfig(
        active_champion_id=governance_raw.get("active_champion_id", "seed_balance_strategy"),
        champion_source=governance_raw.get("champion_source", "manual"),
        manual_effective_date=governance_raw.get("manual_effective_date"),
        manual_reason=governance_raw.get("manual_reason"),
        challengers=tuple(governance_raw.get("challengers", [
            "balance_strategy_v1",
            "positive_strategy_v1",
            "value_strategy_v1",
        ])),
    )

    challenger_weight_raw = cfg.get("challenger_weights", {})
    challenger_cfg = ChallengerConfig(
        positive_growth_weight=float(challenger_weight_raw.get("positive_growth_weight", 0.7)),
        positive_frontier_weight=float(challenger_weight_raw.get("positive_frontier_weight", 0.3)),
    )

    seed_raw = cfg.get("seed_balance_strategy", {})
    selector_cfg = SelectionConfig(
        model_type=seed_raw.get("model_type", "lgbm"),
        label_horizons=tuple(seed_raw.get("label_horizons", [20, 60, 120])),
        label_weights=tuple(seed_raw.get("label_weights", [0.5, 0.3, 0.2])),
        risk_adjusted=bool(seed_raw.get("risk_adjusted", True)),
        industry_col=seed_raw.get("industry_col", "industry_l1"),
    )

    seed_strategy = SeedBalanceStrategy(selector_config=selector_cfg)
    data_dir = args.data_dir or str(get_tushare_root())
    challenger_strategies = MultiAlphaChallengerStrategies(
        data_dir=data_dir,
        config=challenger_cfg,
    )
    engine = ChampionChallengerEngine(
        governance_config=governance,
        seed_strategy=seed_strategy,
        challenger_strategies=challenger_strategies,
    )

    seed_data = None
    if args.seed_input:
        seed_input_path = Path(args.seed_input)
        if not seed_input_path.is_absolute():
            seed_input_path = ROOT / seed_input_path
        seed_data = _load_seed_input(seed_input_path)

    active_champion_id = args.active_champion_id
    if active_champion_id:
        active_champion_id = normalize_strategy_id(active_champion_id)

    result = engine.run(
        trade_date=args.trade_date,
        top_n=args.top_n,
        seed_data=seed_data,
        active_champion_id=active_champion_id,
        allocation_method=args.allocation_method,
        regime=args.regime,
    )

    output_root = Path(args.output_root) if args.output_root else get_data_path("features", "stock_selector", ensure=True)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    saved = save_strategy_outputs(
        output_root=output_root,
        trade_date=args.trade_date,
        champion_id=result["active_champion_id"],
        champion_signals=result["champion_signals"],
        challenger_signals=result["challenger_signals"],
    )

    summary = {
        "trade_date": args.trade_date,
        "active_champion_id": result["active_champion_id"],
        "top_n": args.top_n,
        "allocation_method": args.allocation_method,
        "regime": args.regime,
        "saved_files": {k: str(v) for k, v in saved.items()},
    }
    summary_path = output_root / f"governance_summary_{args.trade_date}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"策略输出已保存: {summary_path}")


if __name__ == "__main__":
    main()
