#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data._shared.runtime import get_data_path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="行业信号质量检查")
    parser.add_argument("--northbound-min-rows", type=int, default=20)
    parser.add_argument("--northbound-max-stale-days", type=int, default=7)
    parser.add_argument("--concept-min-coverage", type=float, default=0.95)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    summary_path = get_data_path("signals", "industry", "industry_signal_contract_summary.json")
    concept_report_path = get_data_path("processed", "concepts", "concept_industry_mapping_report.json")
    summary = _load_json(summary_path)
    concept_report = _load_json(concept_report_path)

    northbound = (summary.get("signal_freshness") or {}).get("northbound_ratio") or {}
    northbound_rows = int(northbound.get("rows", 0) or 0)
    northbound_stale = int(northbound.get("max_stale_days", 99999) or 99999)
    concept_coverage = float(concept_report.get("coverage_rate", 0.0) or 0.0)

    checks = {
        "northbound_rows_ok": northbound_rows >= int(args.northbound_min_rows),
        "northbound_stale_ok": northbound_stale <= int(args.northbound_max_stale_days),
        "concept_coverage_ok": concept_coverage >= float(args.concept_min_coverage),
    }
    passed = all(checks.values())
    payload = {
        "passed": passed,
        "checks": checks,
        "metrics": {
            "northbound_rows": northbound_rows,
            "northbound_max_stale_days": northbound_stale,
            "concept_coverage_rate": concept_coverage,
        },
        "thresholds": {
            "northbound_min_rows": int(args.northbound_min_rows),
            "northbound_max_stale_days": int(args.northbound_max_stale_days),
            "concept_min_coverage": float(args.concept_min_coverage),
        },
    }

    output_path = Path(args.output) if args.output else get_data_path("signals", "industry", "industry_signal_quality_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"质量报告已保存: {output_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
