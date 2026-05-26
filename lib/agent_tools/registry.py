from typing import Any, Dict, List, Optional

from .tool_base import BaseTool, ToolResult, make_tool_schema


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def schema_list(self, fmt: str = "openai") -> List[Dict[str, Any]]:
        return [make_tool_schema(t, fmt=fmt) for t in self._tools.values()]

    def call(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult.fail(name, f"Unknown tool: {name}", params=kwargs)
        return tool(**kwargs)
