#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sage_core.execution import (
    build_orders_from_portfolio,
    create_broker_adapter,
    load_broker_config,
    save_submit_payload,
)


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    if not files:
        return None
    return files[-1]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="券商执行入口（当前支持 PingAn dry-run）")
    parser.add_argument("--broker", type=str, default="pingan", help="券商标识，默认 pingan")
    parser.add_argument("--portfolio-path", type=str, default=None, help="组合文件路径，默认读取最新 portfolio_*.csv")
    parser.add_argument("--execution-context-path", type=str, default=None, help="执行上下文路径，默认读取最新 execution_context_*.json")
    parser.add_argument("--config-path", type=str, default=None, help="券商配置文件路径，默认 config/app/broker.yaml")
    parser.add_argument("--top-n", type=int, default=10, help="下单股票数上限，默认10")
    parser.add_argument("--submit", action="store_true", help="发起实盘提交（当前 PingAn 未实现，默认 dry-run）")
    parser.add_argument("--output-path", type=str, default=None, help="执行回执输出路径")
    args = parser.parse_args()

    portfolio_dir = PROJECT_ROOT / "data" / "portfolio"
    portfolio_path = Path(args.portfolio_path) if args.portfolio_path else _latest_file(portfolio_dir, "portfolio_*.csv")
    if not portfolio_path or not portfolio_path.exists():
        raise FileNotFoundError("未找到组合文件，请先运行 weekly 流程生成 portfolio_*.csv")

    context_path = Path(args.execution_context_path) if args.execution_context_path else _latest_file(
        portfolio_dir, "execution_context_*.json"
    )
    context = _load_json(context_path) if context_path else {}

    portfolio = pd.read_csv(portfolio_path)
    orders = build_orders_from_portfolio(portfolio, top_n=args.top_n)
    if not orders:
        raise ValueError("组合为空，未生成任何订单")

    broker_cfg = load_broker_config(args.config_path)
    brokers = broker_cfg.get("brokers", {}) if isinstance(broker_cfg, dict) else {}
    adapter = create_broker_adapter(args.broker, config=brokers.get(args.broker, {}))
    result = adapter.submit_orders(orders=orders, dry_run=not args.submit)

    output_path = Path(args.output_path) if args.output_path else (
        portfolio_dir / f"broker_submit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "broker": args.broker,
        "submit_mode": "submit" if args.submit else "dry_run",
        "portfolio_path": str(portfolio_path),
        "execution_context_path": str(context_path) if context_path else None,
        "active_champion_id": context.get("active_champion_id"),
        "trade_date": context.get("trade_date"),
        "orders": [order.__dict__ for order in orders],
        "result": result.to_dict(),
    }
    saved = save_submit_payload(output_path, payload)
    print(f"券商执行结果已保存: {saved}")
    print(f"broker={args.broker}, mode={'submit' if args.submit else 'dry_run'}, orders={len(orders)}")


if __name__ == "__main__":
    main()
