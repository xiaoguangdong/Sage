from __future__ import annotations

from typing import Any

from sage.core.types import BaseModel, ModelOutput


class RuleBasedModel(BaseModel):
    """
    Base class for rule-based models.
    """

    def fit(self, dataset: Any) -> None:
        # Rule-based models usually do not require training.
        return None

    def predict(self, dataset: Any) -> ModelOutput:  # pragma: no cover
        raise NotImplementedError

