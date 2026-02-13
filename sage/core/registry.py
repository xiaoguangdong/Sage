from __future__ import annotations

from typing import Any, Callable, Dict, Iterable


class Registry:
    """
    Simple registry to support replaceable modules.

    Example:
        registry = Registry()

        @registry.register("trend_rule")
        class TrendRuleModel: ...
    """

    def __init__(self, name: str = "registry") -> None:
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, key: str) -> Callable[[Any], Any]:
        def _decorator(obj: Any) -> Any:
            self._items[key] = obj
            return obj

        return _decorator

    def get(self, key: str) -> Any:
        if key not in self._items:
            raise KeyError(f"{key} not found in {self.name}")
        return self._items[key]

    def list(self) -> Iterable[str]:
        return sorted(self._items.keys())

