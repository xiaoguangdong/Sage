from __future__ import annotations

from typing import Any

from sage.core.types import ModelOutput
from sage.models.base import RuleBasedModel


class MacroScenarioModel(RuleBasedModel):
    """
    Placeholder for macro scenario classification.

    Expected to output industry scenario scores and labels:
      - RECOVERY / BOOM / RECESSION / NEUTRAL
    """

    name = "macro_scenario_rule"

    def predict(self, dataset: Any) -> ModelOutput:
        raise NotImplementedError("MacroScenarioModel is not implemented yet.")

