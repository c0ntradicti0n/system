import os
import re
import shutil
import time
from pathlib import Path

from .tool_base import BaseTool, ToolResult

TOPIC_FILE = re.compile(r"^\.[^.].*\.md$")
ANTONYM_FILE = re.compile(r"^_[^_].*\.md$")


def _find_topic_file(dir_path: str) -> str | None:
    try:
        for fname in os.listdir(dir_path):
            if TOPIC_FILE.match(fname):
                return fname
    except OSError:
        pass
    return None


def _find_antonym_file(dir_path: str) -> str | None:
    try:
        for fname in os.listdir(dir_path):
            if ANTONYM_FILE.match(fname):
                return fname
    except OSError:
        pass
    return None


def _get_content(path: str) -> str:
    try:
        return Path(path).read_text("utf-8")
    except Exception:
        return ""


def _collect_triple_keys(dir_path: str) -> list[str]:
    """Collect child triple directories (1,2,3) that exist."""
    keys = []
    try:
        for name in sorted(os.listdir(dir_path)):
            if name in {"1", "2", "3"} and os.path.isdir(os.path.join(dir_path, name)):
                keys.append(name)
    except OSError:
        pass
    return keys


def _merge_recursive(keep_dir: str, remove_dir: str, merge_log: list, depth: int = 0):
    """Recursively merge content from remove_dir into keep_dir."""
    indent = "  " * depth

    # Merge topic file
    keep_topic = _find_topic_file(keep_dir)
    remove_topic = _find_topic_file(remove_dir)
    if remove_topic and not keep_topic:
        src = os.path.join(remove_dir, remove_topic)
        dst = os.path.join(keep_dir, remove_topic)
        shutil.copy2(src, dst)
        merge_log.append({"type": "copied_topic", "from": src, "to": dst})

    # Merge antonym file
    keep_ant = _find_antonym_file(keep_dir)
    remove_ant = _find_antonym_file(remove_dir)
    if remove_ant:
        src = os.path.join(remove_dir, remove_ant)
        dst = os.path.join(keep_dir, remove_ant)
        if not keep_ant:
            shutil.copy2(src, dst)
            merge_log.append({"type": "copied_antonym", "from": src, "to": dst})
        else:
            # Append content to existing antonym
            existing = _get_content(dst)
            new = _get_content(src)
            if new and new not in existing:
                with open(dst, "a") as f:
                    f.write(f"\n\n--- Merged from {os.path.basename(remove_dir)} ---\n")
                    f.write(new)
                merge_log.append({"type": "appended_antonym", "from": src, "to": dst})

    # Merge child triples
    remove_keys = _collect_triple_keys(remove_dir)
    for key in remove_keys:
        keep_child = os.path.join(keep_dir, key)
        remove_child = os.path.join(remove_dir, key)
        if not os.path.exists(keep_child):
            os.makedirs(os.path.dirname(keep_child), exist_ok=True)
            shutil.copytree(remove_child, keep_child)
            merge_log.append({"type": "copied_triple", "from": remove_child, "to": keep_child})
        else:
            _merge_recursive(keep_child, remove_child, merge_log, depth + 1)


class MergeTopicsTool(BaseTool):
    name = "merge_topics"
    description = "Merge two topic directories into one. Preserves content from both directories by copying missing files, appending antonym content, and recursively merging child triples. Then removes the source directory. Use this to consolidate scattered concepts (e.g. music appearing in 3 branches) into a single canonical location."
    parameters = {
        "type": "object",
        "properties": {
            "basepath": {"type": "string", "description": "Root path of the dialectics tree"},
            "keep_path": {"type": "string", "description": "Path to keep (topic will survive, e.g. '3/1/1/2/.../2/2'). This directory must exist."},
            "remove_path": {"type": "string", "description": "Path to remove (topic will be merged into keep, then deleted, e.g. '2/3/2/3/2/2'). This directory must exist."},
            "dry_run": {"type": "boolean", "description": "If true, only report what would merge and delete", "default": True},
            "commit": {"type": "boolean", "description": "If false, merge files but leave git changes unstaged. Only used when dry_run=False.", "default": True},
        },
        "required": ["basepath", "keep_path", "remove_path"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        basepath = kwargs["basepath"]
        keep_path = kwargs["keep_path"]
        remove_path = kwargs["remove_path"]
        dry_run = kwargs.get("dry_run", True)
        commit = kwargs.get("commit", True)

        keep_dir = os.path.join(basepath, keep_path)
        remove_dir = os.path.join(basepath, remove_path)

        if not os.path.isdir(keep_dir):
            return ToolResult.fail(self.name, f"Keep directory not found: {keep_dir}", params=kwargs, duration_s=time.time() - t0)
        if not os.path.isdir(remove_dir):
            return ToolResult.fail(self.name, f"Remove directory not found: {remove_dir}", params=kwargs, duration_s=time.time() - t0)
        if keep_path == remove_path:
            return ToolResult.fail(self.name, "keep_path and remove_path are the same", params=kwargs, duration_s=time.time() - t0)

        try:
            keep_topic = _find_topic_file(keep_dir)
            remove_topic = _find_topic_file(remove_dir)

            merge_log = []

            # Dry-run: plan what would happen
            plan = {
                "keep_path": keep_path,
                "remove_path": remove_path,
                "keep_topic": keep_topic,
                "remove_topic": remove_topic,
                "keep_antonym": _find_antonym_file(keep_dir),
                "remove_antonym": _find_antonym_file(remove_dir),
                "keep_children": _collect_triple_keys(keep_dir),
                "remove_children": _collect_triple_keys(remove_dir),
            }

            # What will be copied
            plan["will_copy_topic"] = bool(remove_topic and not keep_topic)
            plan["will_copy_antonym"] = bool(_find_antonym_file(remove_dir) and not _find_antonym_file(keep_dir))
            plan["will_merge_children"] = _collect_triple_keys(remove_dir)
            plan["will_delete"] = remove_path

            if not dry_run:
                _merge_recursive(keep_dir, remove_dir, merge_log)

                # Remove the source directory
                shutil.rmtree(remove_dir)
                merge_log.append({"type": "removed_directory", "path": remove_path})

                # Git commit (only if commit=True)
                if commit:
                    try:
                        from lib.os_tools import git_auto_commit
                        with git_auto_commit(basepath, f"merge_topics {remove_path} -> {keep_path}"):
                            pass
                    except Exception as e:
                        merge_log.append({"type": "git_warning", "detail": str(e)})

            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "plan": plan,
                "merge_log": merge_log if not dry_run else [],
                "actions_performed": len(merge_log),
                "dry_run": dry_run,
                "committed": commit and not dry_run,
            }, params=kwargs, duration_s=duration)

        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
