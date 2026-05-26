import time

from .tool_base import BaseTool, ToolResult


class WriteTocToFilesystemTool(BaseTool):
    name = "write_toc_to_filesystem"
    description = "Write a nested tree dict to the filesystem as .md files. Creates directory structure matching fractal triple keys."
    parameters = {
        "type": "object",
        "properties": {
            "tree_json": {"type": "object", "description": "Nested tree dict to write"},
            "base_path": {"type": "string", "description": "Target filesystem path"},
            "prefix_items": {"type": "boolean", "description": "Add key prefixes to filenames", "default": True},
        },
        "required": ["tree_json", "base_path"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            from lib.helper import nested_dict_to_filesystem
            creations = nested_dict_to_filesystem(
                kwargs["base_path"],
                kwargs["tree_json"],
                prefix_items=kwargs.get("prefix_items", True),
            )
            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "files_created": creations,
                "count": len(creations),
            }, params=kwargs, duration_s=duration)
        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)


class SanitizeTocTool(BaseTool):
    name = "sanitize_toc"
    description = "Sanitize a tree dict: fix key formatting, remove bad chars, enforce healthy key patterns for the fractal system."
    parameters = {
        "type": "object",
        "properties": {
            "tree_json": {"type": "object", "description": "Nested tree dict to sanitize"},
        },
        "required": ["tree_json"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            from lib.helper import sanitize_nested_dict
            result = sanitize_nested_dict(
                kwargs["tree_json"],
                sanitize_key=lambda x: str(x).replace("..", ".").replace("__", "_").replace("/", "-") if isinstance(x, str) else x,
            )
            duration = time.time() - t0
            return ToolResult.ok(self.name, result, params=kwargs, duration_s=duration)
        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
