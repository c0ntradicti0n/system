import time

from lib.helper import tree as _tree, OutputLevel

from .tool_base import BaseTool, ToolResult


class ScanFilesystemTool(BaseTool):
    name = "scan_filesystem"
    description = "Scan dialectical triangle filesystem. Returns nested dict mirroring directory tree. Keys: '.' = topic title, '1/2/3' = thesis/antithesis/synthesis, '_' = antonym relation."
    parameters = {
        "type": "object",
        "properties": {
            "startpath": {"type": "string", "description": "Relative path with / separator (e.g. '1/3/1' for topic 131, '2/1/3' for topic 213, empty string '' for root)"},
            "basepath": {"type": "string", "description": "Root base path (default: config.system_path)"},
            "depth": {"type": "integer", "description": "Max recursion depth", "default": 2},
            "sparse": {"type": "boolean", "description": "Truncate file content to 100 chars", "default": True},
            "info_radius": {"type": "integer", "description": "Distance-based visibility radius", "default": 100},
            "location": {"type": "string", "description": "Focus location for distance pruning", "default": ""},
            "output_level": {
                "type": "string",
                "enum": ["FILE", "FILENAMES", "DIRECTORY", "TOPIC"],
                "description": "FILE=full content, FILENAMES=names only, DIRECTORY=dirs only, TOPIC=dirs with . prefix",
                "default": "FILENAMES",
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Items to exclude",
                "default": [".git", ".git.md", ".idea"],
            },
            "prefix_items": {"type": "boolean", "description": "Prefix filenames with key char", "default": True},
        },
        "required": ["startpath"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        try:
            from lib import config
            basepath = kwargs.get("basepath", config.system_path)
            level_map = {
                "FILE": OutputLevel.FILE,
                "FILENAMES": OutputLevel.FILENAMES,
                "DIRECTORY": OutputLevel.DIRECTORY,
                "TOPIC": OutputLevel.TOPIC,
                None: None,
            }
            out_level = level_map.get(kwargs.get("output_level"))
            result = _tree(
                basepath=basepath,
                startpath=kwargs["startpath"],
                depth=kwargs.get("depth", 2),
                sparse=kwargs.get("sparse", True),
                info_radius=kwargs.get("info_radius", 100),
                location=kwargs.get("location", ""),
                pre_set_output_level=out_level,
                exclude=kwargs.get("exclude", [".git", ".git.md", ".idea"]),
                prefix_items=kwargs.get("prefix_items", True),
            )
            duration = time.time() - t0
            summary = f"Scanned {len(result)} top-level keys"
            return ToolResult.ok(self.name, result, params=kwargs, duration_s=duration)
        except FileNotFoundError as e:
            return ToolResult.fail(self.name, f"Path not found: {e}", params=kwargs, duration_s=time.time() - t0)
        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
