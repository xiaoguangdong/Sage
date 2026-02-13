from __future__ import annotations

from typing import Any

from sage.core.types import ModelOutput
from sage.models.base import RuleBasedModel


class EntryFilterModel(RuleBasedModel):
    """
    Placeholder for entry/exit filtering (rule + light ML).
    """

    name = "entry_filter"

    def predict(self, dataset: Any) -> ModelOutput:
        raise NotImplementedError("EntryFilterModel is not implemented yet.")

