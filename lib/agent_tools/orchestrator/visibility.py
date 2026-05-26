import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class OperationLog:
    tool_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    duration_s: float = 0.0
    timestamp: str = ""
    status: str = "pending"  # pending | running | success | error

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class VisibilityLogger:
    def __init__(self, history_file: Optional[str] = None):
        self.history: List[OperationLog] = []
        self.history_file = history_file
        self._logger = logging.getLogger("agent_tools")

    def log_start(self, tool_name: str, params: Dict[str, Any]) -> int:
        op = OperationLog(tool_name=tool_name, params=params, status="running")
        self.history.append(op)
        step = len(self.history)
        self._logger.info("[%d] %s %s", step, tool_name, _summarize_params(params))
        return step - 1

    def log_success(self, idx: int, result_summary: str, duration_s: float) -> None:
        if 0 <= idx < len(self.history):
            op = self.history[idx]
            op.status = "success"
            op.result_summary = result_summary
            op.duration_s = duration_s
            self._logger.info(
                "  -> ok (%.2fs) %s", duration_s, _truncate(result_summary, 120)
            )
            self._append_file(op)

    def log_error(self, idx: int, error: str, duration_s: float) -> None:
        if 0 <= idx < len(self.history):
            op = self.history[idx]
            op.status = "error"
            op.result_summary = error
            op.duration_s = duration_s
            self._logger.error("  -> fail (%.2fs) %s", duration_s, error)
            self._append_file(op)

    def get_history(self) -> List[OperationLog]:
        return self.history

    def _append_file(self, op: OperationLog) -> None:
        if not self.history_file:
            return
        try:
            with open(self.history_file, "a") as f:
                f.write(json.dumps(asdict(op)) + "\n")
        except OSError:
            pass


def _summarize_params(params: Dict[str, Any]) -> str:
    parts = []
    for k, v in params.items():
        if isinstance(v, str) and len(v) > 60:
            parts.append(f"{k}={v[:60]}...")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def _truncate(s: str, n: int) -> str:
    return s[:n] + "..." if len(s) > n else s
