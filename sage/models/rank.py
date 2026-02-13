from __future__ import annotations

from typing import Any

from sage.core.types import ModelOutput
from sage.models.base import RuleBasedModel


class CrossSectionRankModel(RuleBasedModel):
    """
    Placeholder for cross-sectional ranking model.
    """

    name = "rank_model"

    def predict(self, dataset: Any) -> ModelOutput:
        raise NotImplementedError("CrossSectionRankModel is not implemented yet.")

