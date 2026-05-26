from dataclasses import dataclass, field
from typing import Any, Dict, List

from .visibility import OperationLog


@dataclass
class AgentState:
    intent: str = ""
    history: List[OperationLog] = field(default_factory=list)
    accumulated_context: Dict[str, Any] = field(default_factory=dict)
    step_number: int = 0

    def add_result(self, key: str, value: Any) -> None:
        self.accumulated_context[key] = value

    def get_result(self, key: str, default=None) -> Any:
        return self.accumulated_context.get(key, default)
