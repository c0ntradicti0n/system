import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


@dataclass
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_s: float = 0.0
    tool_name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, tool_name, data, params=None, duration_s=0.0):
        return cls(
            success=True, data=data, tool_name=tool_name,
            params=params or {}, duration_s=duration_s,
        )

    @classmethod
    def fail(cls, tool_name, error, params=None, duration_s=0.0):
        return cls(
            success=False, error=str(error), tool_name=tool_name,
            params=params or {}, duration_s=duration_s,
        )


class BaseTool(Protocol):
    name: str
    description: str
    parameters: Dict[str, Any]

    def __call__(self, **kwargs) -> ToolResult:
        ...


def make_tool_schema(tool: BaseTool, fmt: str = "openai") -> Dict[str, Any]:
    """Generate tool schema for LLM tool-use API.
    fmt='openai': {'type':'function','function':{name,description,parameters}}
    fmt='anthropic': {'name':, 'description':, 'input_schema':}
    """
    if fmt == "anthropic":
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }
    # OpenAI / DeepSeek format
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
