from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BrokerOrder:
    ts_code: str
    target_weight: float
    side: str = "TARGET"
    note: str = ""


@dataclass
class BrokerSubmitResult:
    broker: str
    dry_run: bool
    submitted_at: str
    accepted_orders: int
    rejected_orders: int
    message: str
    details: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_ts_code(code: str) -> str:
    value = str(code).strip().upper()
    if not value:
        return value
    if "." in value:
        return value
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) != 6:
        return value
    if digits.startswith(("6", "9")):
        return f"{digits}.SH"
    if digits.startswith(("8", "4")):
        return f"{digits}.BJ"
    return f"{digits}.SZ"


def build_orders_from_portfolio(portfolio: pd.DataFrame, top_n: int | None = None) -> List[BrokerOrder]:
    if portfolio is None or portfolio.empty:
        return []
    frame = portfolio.copy()
    if "ts_code" not in frame.columns and "code" in frame.columns:
        frame["ts_code"] = frame["code"].apply(normalize_ts_code)
    if "target_weight" not in frame.columns and "weight" in frame.columns:
        frame["target_weight"] = pd.to_numeric(frame["weight"], errors="coerce")
    required = {"ts_code", "target_weight"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"portfolio 缺少字段: {missing}")

    frame["ts_code"] = frame["ts_code"].astype(str).apply(normalize_ts_code)
    frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce")
    frame = frame.dropna(subset=["ts_code", "target_weight"])
    frame = frame[frame["target_weight"] > 0].copy()
    frame = frame.sort_values("target_weight", ascending=False).drop_duplicates(subset=["ts_code"], keep="first")
    if top_n and top_n > 0:
        frame = frame.head(int(top_n))

    return [
        BrokerOrder(
            ts_code=row.ts_code,
            target_weight=float(row.target_weight),
            side="TARGET",
            note="from_portfolio",
        )
        for row in frame.itertuples(index=False)
    ]


class PingAnSecuritiesAdapter:
    broker_name = "pingan"

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.endpoint = str(self.config.get("endpoint", "")).strip()
        self.client_id = str(self.config.get("client_id", "")).strip()
        self.account_id = str(self.config.get("account_id", "")).strip()

    def submit_orders(self, orders: Sequence[BrokerOrder], dry_run: bool = True) -> BrokerSubmitResult:
        details = [asdict(order) for order in orders]
        if dry_run:
            return BrokerSubmitResult(
                broker=self.broker_name,
                dry_run=True,
                submitted_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                accepted_orders=len(details),
                rejected_orders=0,
                message="PingAn 入口已接入（dry-run），实盘API待后续实现",
                details=details,
            )
        raise NotImplementedError(
            "PingAn 实盘API尚未接入。当前只支持 dry-run。后续需补齐鉴权、下单、撤单、回报与持仓对账。"
        )


def create_broker_adapter(broker: str, config: Dict[str, Any] | None = None) -> PingAnSecuritiesAdapter:
    broker_key = str(broker).strip().lower()
    if broker_key in {"pingan", "pingan_securities"}:
        return PingAnSecuritiesAdapter(config=config)
    raise ValueError(f"不支持的券商类型: {broker}")


def load_broker_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else (PROJECT_ROOT / "config" / "app" / "broker.yaml")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def save_submit_payload(path: str | Path, payload: Dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
