import os
import re
import time
from pathlib import Path

from .tool_base import BaseTool, ToolResult

# Match dot-prefixed topic files like .Music.md or .Space-Time.md
TOPIC_FILE = re.compile(r"^\.[^.].*\.md$")


def _find_topic_file(dir_path: str) -> str | None:
    """Find the dot-prefixed topic file in a directory."""
    try:
        for fname in os.listdir(dir_path):
            if TOPIC_FILE.match(fname):
                return fname
    except OSError:
        pass
    return None


def _title_to_filename(title: str) -> str:
    """Convert a title string to a safe filename fragment."""
    safe = re.sub(r'["\':;!@#$%^&*()+={}\[\]|\\,<>?~`]', "", title)
    safe = safe.strip().replace(" ", "-")
    return f".{safe}.md"


class RenameTopicTool(BaseTool):
    name = "rename_topic"
    description = "Rename a topic file inside a directory. Changes the dot-prefixed topic title file (e.g. .Music.md -> .Music-as-Discipline.md). Does NOT move the directory, only renames the topic marker file. The antonym file is left unchanged since it has its own distinct title. Use this to disambiguate scattered concepts or fix inconsistent naming."
    parameters = {
        "type": "object",
        "properties": {
            "basepath": {"type": "string", "description": "Root path of the dialectics tree"},
            "path": {"type": "string", "description": "Path with / separator (e.g. '1/2/3' for topic 123)"},
            "new_title": {"type": "string", "description": "New title for the topic (e.g. 'Music-as-Discipline'). File will become .New-Title.md"},
            "dry_run": {"type": "boolean", "description": "If true, only report what would change", "default": True},
            "commit": {"type": "boolean", "description": "If false, rename file but leave git changes unstaged for review. Only used when dry_run=False.", "default": True},
        },
        "required": ["basepath", "path", "new_title"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        basepath = kwargs["basepath"]
        path_key = kwargs["path"]
        new_title = kwargs["new_title"]
        dry_run = kwargs.get("dry_run", True)
        commit = kwargs.get("commit", True)

        dir_path = os.path.join(basepath, path_key)
        if not os.path.isdir(dir_path):
            return ToolResult.fail(self.name, f"Directory not found: {dir_path}", params=kwargs, duration_s=time.time() - t0)

        try:
            old_topic = _find_topic_file(dir_path)

            if not old_topic:
                return ToolResult.fail(self.name, f"No dot-prefixed topic file found in {dir_path}", params=kwargs, duration_s=time.time() - t0)

            changes = []

            new_topic = _title_to_filename(new_title)
            if old_topic == new_topic:
                changes.append({"type": "no_change", "file": old_topic, "detail": "Already has this title"})
            else:
                changes.append({
                    "type": "rename_topic",
                    "from": old_topic,
                    "to": new_topic,
                    "path": path_key,
                })

            if not dry_run:
                for change in changes:
                    if change["type"] == "no_change":
                        continue
                    old_path = os.path.join(dir_path, change["from"])
                    new_path = os.path.join(dir_path, change["to"])
                    if os.path.exists(old_path):
                        os.rename(old_path, new_path)

                # Git commit (only if commit=True)
                if commit:
                    try:
                        from lib.os_tools import git_auto_commit
                        with git_auto_commit(basepath, f"rename_topic {path_key}: {new_title}"):
                            pass
                    except Exception as e:
                        changes.append({"type": "git_warning", "detail": str(e)})

            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "path": path_key,
                "new_title": new_title,
                "changes": changes,
                "change_count": len([c for c in changes if c["type"] != "no_change"]),
                "dry_run": dry_run,
                "committed": commit and not dry_run,
            }, params=kwargs, duration_s=duration)

        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
