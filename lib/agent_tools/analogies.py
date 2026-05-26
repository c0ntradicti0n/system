import time

from .tool_base import BaseTool, ToolResult


class FindAnalogiesTool(BaseTool):
    name = "find_analogies"
    description = "Find analogous paths across levels of the fractal tree using the system's analogy mapping."
    parameters = {
        "type": "object",
        "properties": {
            "tree_json": {"type": "object", "description": "Serialized tree dict"},
            "pivot_path": {"type": "string", "description": "Path to find analogies for (e.g. '131')"},
        },
        "required": ["tree_json", "pivot_path"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            from lib.helper import get_analogue_from_nested_dict
            pivot = list(kwargs["pivot_path"])
            results = get_analogue_from_nested_dict(kwargs["tree_json"], pivot)
            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "analogies": [
                    {"path": "".join(str(p) for p in path), "value": val}
                    for path, val in results if val
                ],
            }, params=kwargs, duration_s=duration)
        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
