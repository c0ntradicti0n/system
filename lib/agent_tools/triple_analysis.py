import json
import time

from ._path import _project_root
from .tool_base import BaseTool, ToolResult


class AnalyseTriplesTool(BaseTool):
    name = "analyse_triples"
    description = "Analyse the fractal tree for incomplete triples. Returns shallowest and deepest paths that are missing 1/2/3/_ entries."
    parameters = {
        "type": "object",
        "properties": {
            "tree_json": {
                "type": "object",
                "description": "Serialized tree dict (output from scan_filesystem)",
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Path prefixes to exclude from analysis",
                "default": ["111"],
            },
        },
        "required": ["tree_json"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            from philo.analyser import analyse_toc
            exclude = kwargs.get("exclude", ["111"])
            result = analyse_toc(kwargs["tree_json"], exclude=exclude)
            duration = time.time() - t0
            summary = f"{len(result['min_depth_paths'])} shallow, {len(result['max_depth_paths'])} deep incomplete paths"
            return ToolResult.ok(self.name, result, params=kwargs, duration_s=duration)
        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)


class MissingContentTool(BaseTool):
    name = "find_missing_content"
    description = "Find paths in the fractal tree that have no text content. Returns paths missing content with their before-context."
    parameters = {
        "type": "object",
        "properties": {
            "tree_json": {"type": "object", "description": "Serialized tree dict"},
            "base_path": {"type": "string", "description": "Filesystem base path for content files"},
            "exclude": {
                "type": "array", "items": {"type": "string"},
                "default": ["111"],
            },
        },
        "required": ["tree_json", "base_path"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            from philo.analyser import without_text
            results = []
            for paths, (before_path, before_content) in without_text(
                kwargs["tree_json"], kwargs["base_path"], exclude=kwargs.get("exclude", ["111"])
            ):
                results.append({
                    "missing_paths": ["".join(str(p) for p in path) for path in paths],
                    "before_path": "".join(str(p) for p in before_path) if before_path else None,
                    "before_content": before_content[:200] if before_content else None,
                })
            duration = time.time() - t0
            return ToolResult.ok(self.name, results, params=kwargs, duration_s=duration)
        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
