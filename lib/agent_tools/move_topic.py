import os
import time

from .tool_base import BaseTool, ToolResult


def _validate_triple_path(path: str) -> bool:
    """Check if path consists only of valid triple components (1,2,3,_,.)."""
    parts = path.strip("/").split("/")
    return all(p in {"1", "2", "3", "_", "."} for p in parts)


class MoveTopicTool(BaseTool):
    name = "move_topic"
    description = "Move an entire triple directory to a new location. Relocates all content including topic file, antonym file, and subtriples. If target already exists, fails with a message suggesting merge_topic instead. Use this to reorganize misplaced concepts across the dialectical tree."
    parameters = {
        "type": "object",
        "properties": {
            "basepath": {"type": "string", "description": "Root path of the dialectics tree"},
            "source_path": {"type": "string", "description": "Source path with / separator (e.g. '2/3/2/3/2/2')"},
            "target_path": {"type": "string", "description": "Target path with / separator (e.g. '3/1/4'). Must not already exist."},
            "dry_run": {"type": "boolean", "description": "If true, only report what would move", "default": True},
            "commit": {"type": "boolean", "description": "If false, move files but leave git changes unstaged. Only used when dry_run=False.", "default": True},
        },
        "required": ["basepath", "source_path", "target_path"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        basepath = kwargs["basepath"]
        source_path = kwargs["source_path"]
        target_path = kwargs["target_path"]
        dry_run = kwargs.get("dry_run", True)
        commit = kwargs.get("commit", True)

        source_dir = os.path.join(basepath, source_path)
        target_dir = os.path.join(basepath, target_path)

        if not os.path.isdir(source_dir):
            return ToolResult.fail(self.name, f"Source directory not found: {source_dir}", params=kwargs, duration_s=time.time() - t0)

        if os.path.exists(target_dir):
            return ToolResult.fail(self.name, f"Target already exists: {target_dir}. Use merge_topic instead.", params=kwargs, duration_s=time.time() - t0)

        try:
            # Collect info about what will move
            file_count = 0
            dir_count = 0
            for dirpath, dirnames, filenames in os.walk(source_dir):
                rel = os.path.relpath(dirpath, source_dir)
                if rel == ".":
                    rel = ""
                entry_count = {"files": len(filenames), "dirs": len(dirnames)}
                file_count += entry_count["files"]
                dir_count += entry_count["dirs"]

            move_plan = {
                "source": source_path,
                "target": target_path,
                "total_files": file_count,
                "total_dirs": dir_count,
                "source_exists": os.path.isdir(source_dir),
                "target_exists": False,
                "committed": commit and not dry_run,
            }

            if not dry_run:
                os.makedirs(os.path.dirname(target_dir), exist_ok=True)
                os.rename(source_dir, target_dir)

                # Git commit (only if commit=True)
                if commit:
                    try:
                        from lib.os_tools import git_auto_commit
                        with git_auto_commit(basepath, f"move_topic {source_path} -> {target_path}"):
                            pass
                    except Exception as e:
                        move_plan["git_warning"] = str(e)

            duration = time.time() - t0
            return ToolResult.ok(self.name, move_plan, params=kwargs, duration_s=duration)

        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
