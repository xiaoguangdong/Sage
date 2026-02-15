import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sage_core.models.stock_selector import SelectionConfig
from sage_core.models.strategy_governance import (
    ChampionChallengerEngine,
    MultiAlphaChallengerStrategies,
    SeedBalanceStrategy,
    StrategyGovernanceConfig,
    decide_auto_promotion,
    normalize_strategy_id,
)


class DummyMultiAlphaSelector:
    def select(self, trade_date: str, top_n: int = 30, allocation_method: str = "fixed", regime: str = "sideways"):
        rows = []
        for i in range(1, 11):
            rows.append({
                "ts_code": f"00000{i}.SZ",
                "trade_date": trade_date,
                "close": 10 + i,
                "value_score": 1.0 / i,
                "growth_score": i / 10.0,
                "frontier_score": (11 - i) / 10.0,
                "combined_score": (1.0 / i) * 0.4 + (i / 10.0) * 0.4 + ((11 - i) / 10.0) * 0.2,
            })
        return {"all_scores": pd.DataFrame(rows)}


class TestStrategyGovernance(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=150, freq="B")
        codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
        industry = {
            "000001.SZ": "801010",
            "000002.SZ": "801010",
            "000003.SZ": "801020",
            "000004.SZ": "801020",
        }

        rows = []
        for code in codes:
            price = 20 + np.cumsum(np.random.randn(len(dates)) * 0.2)
            turnover = np.random.uniform(0.01, 0.2, len(dates))
            for idx, dt in enumerate(dates):
                rows.append({
                    "trade_date": dt.strftime("%Y%m%d"),
                    "ts_code": code,
                    "close": float(price[idx]),
                    "turnover": float(turnover[idx]),
                    "industry_l1": industry[code],
                })
        self.seed_data = pd.DataFrame(rows)
        self.trade_date = self.seed_data["trade_date"].max()

    def test_normalize_strategy_id_alias(self):
        self.assertEqual(normalize_strategy_id("seed_banlance_strategy"), "seed_balance_strategy")
        self.assertEqual(normalize_strategy_id("value_strategy_v1"), "value_strategy_v1")

    def test_seed_balance_strategy_schema(self):
        config = SelectionConfig(
            model_type="rule",
            label_horizons=(5,),
            label_weights=(1.0,),
            industry_col="industry_l1",
        )
        strategy = SeedBalanceStrategy(selector_config=config)
        signals = strategy.generate_signals(self.seed_data, trade_date=self.trade_date, top_n=3)
        self.assertEqual(list(signals.columns), ["trade_date", "ts_code", "score", "rank", "confidence", "model_version"])
        self.assertLessEqual(len(signals), 3)
        self.assertTrue(((signals["confidence"] >= 0) & (signals["confidence"] <= 1)).all())

    def test_challenger_split_three_strategies(self):
        challengers = MultiAlphaChallengerStrategies(selector=DummyMultiAlphaSelector())
        outputs = challengers.generate_signals(trade_date="20260215", top_n=5)
        self.assertEqual(set(outputs.keys()), {"balance_strategy_v1", "positive_strategy_v1", "value_strategy_v1"})
        for frame in outputs.values():
            self.assertEqual(list(frame.columns), ["trade_date", "ts_code", "score", "rank", "confidence", "model_version"])
            self.assertLessEqual(len(frame), 5)

    def test_engine_supports_manual_champion(self):
        config = SelectionConfig(
            model_type="rule",
            label_horizons=(5,),
            label_weights=(1.0,),
            industry_col="industry_l1",
        )
        engine = ChampionChallengerEngine(
            governance_config=StrategyGovernanceConfig(active_champion_id="seed_balance_strategy"),
            seed_strategy=SeedBalanceStrategy(selector_config=config),
            challenger_strategies=MultiAlphaChallengerStrategies(selector=DummyMultiAlphaSelector()),
        )

        # 1) 默认冠军 seed
        result_seed = engine.run(
            trade_date=self.trade_date,
            top_n=3,
            seed_data=self.seed_data,
        )
        self.assertEqual(result_seed["active_champion_id"], "seed_balance_strategy")
        self.assertLessEqual(len(result_seed["champion_signals"]), 3)

        # 2) 手动切换冠军到 challenger
        result_challenger = engine.run(
            trade_date=self.trade_date,
            top_n=3,
            seed_data=self.seed_data,
            active_champion_id="positive_strategy_v1",
        )
        self.assertEqual(result_challenger["active_champion_id"], "positive_strategy_v1")
        self.assertLessEqual(len(result_challenger["champion_signals"]), 3)

    def test_auto_promotion_disabled(self):
        eval_df = pd.DataFrame([
            {
                "strategy_id": "seed_balance_strategy",
                "trade_date": "2026-01-03",
                "cost_return": 0.01,
                "max_drawdown": -0.05,
                "sharpe": 0.8,
                "turnover": 0.2,
            },
            {
                "strategy_id": "positive_strategy_v1",
                "trade_date": "2026-01-03",
                "cost_return": 0.03,
                "max_drawdown": -0.04,
                "sharpe": 1.1,
                "turnover": 0.18,
            },
        ])
        decision = decide_auto_promotion(
            evaluation_df=eval_df,
            current_champion="seed_balance_strategy",
            challengers=("positive_strategy_v1",),
            enabled=False,
        )
        self.assertFalse(decision["promoted"])
        self.assertEqual(decision["reason"], "disabled")

    def test_auto_promotion_blocked_by_manual_mode(self):
        eval_df = pd.DataFrame([
            {
                "strategy_id": "seed_balance_strategy",
                "trade_date": "2026-01-03",
                "cost_return": 0.01,
                "max_drawdown": -0.05,
                "sharpe": 0.8,
                "turnover": 0.2,
            },
            {
                "strategy_id": "positive_strategy_v1",
                "trade_date": "2026-01-03",
                "cost_return": 0.03,
                "max_drawdown": -0.04,
                "sharpe": 1.1,
                "turnover": 0.18,
            },
        ])
        decision = decide_auto_promotion(
            evaluation_df=eval_df,
            current_champion="seed_balance_strategy",
            challengers=("positive_strategy_v1",),
            enabled=True,
            manual_mode=True,
            allow_when_manual=False,
        )
        self.assertFalse(decision["promoted"])
        self.assertEqual(decision["reason"], "manual_mode")

    def test_auto_promotion_passes_with_consecutive_outperformance(self):
        eval_rows = []
        for date, champion_ret, challenger_ret in [
            ("2026-01-03", 0.01, 0.02),
            ("2026-01-10", 0.00, 0.025),
            ("2026-01-17", 0.015, 0.03),
        ]:
            eval_rows.append({
                "strategy_id": "seed_balance_strategy",
                "trade_date": date,
                "cost_return": champion_ret,
                "max_drawdown": -0.05,
                "sharpe": 0.7,
                "turnover": 0.25,
                "data_quality": 1.0,
            })
            eval_rows.append({
                "strategy_id": "positive_strategy_v1",
                "trade_date": date,
                "cost_return": challenger_ret,
                "max_drawdown": -0.04,
                "sharpe": 1.0,
                "turnover": 0.20,
                "data_quality": 1.0,
            })
        eval_df = pd.DataFrame(eval_rows)
        decision = decide_auto_promotion(
            evaluation_df=eval_df,
            current_champion="seed_balance_strategy",
            challengers=("positive_strategy_v1",),
            as_of_date="20260117",
            enabled=True,
            manual_mode=False,
            consecutive_periods=3,
            min_cost_return_diff=0.001,
            min_sharpe_diff=0.05,
        )
        self.assertTrue(decision["promoted"])
        self.assertEqual(decision["next_champion"], "positive_strategy_v1")
        self.assertEqual(decision["reason"], "challenger_outperformed")
        self.assertEqual(decision["periods_checked"], 3)

    def test_auto_promotion_rejected_when_gate_not_met(self):
        eval_rows = []
        for date in ["2026-01-03", "2026-01-10", "2026-01-17"]:
            eval_rows.append({
                "strategy_id": "seed_balance_strategy",
                "trade_date": date,
                "cost_return": 0.02,
                "max_drawdown": -0.05,
                "sharpe": 0.9,
                "turnover": 0.20,
                "data_quality": 1.0,
            })
            eval_rows.append({
                "strategy_id": "positive_strategy_v1",
                "trade_date": date,
                "cost_return": 0.03,
                "max_drawdown": -0.04,
                "sharpe": 1.1,
                "turnover": 0.50,
                "data_quality": 1.0,
            })
        eval_df = pd.DataFrame(eval_rows)
        decision = decide_auto_promotion(
            evaluation_df=eval_df,
            current_champion="seed_balance_strategy",
            challengers=("positive_strategy_v1",),
            enabled=True,
            consecutive_periods=3,
            max_turnover_multiplier=1.1,
        )
        self.assertFalse(decision["promoted"])
        self.assertEqual(decision["next_champion"], "seed_balance_strategy")
        self.assertEqual(decision["reason"], "no_eligible_challenger")


if __name__ == "__main__":
    unittest.main()
