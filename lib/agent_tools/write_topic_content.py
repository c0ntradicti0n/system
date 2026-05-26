import os
import re
import time

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


class WriteTopicContentTool(BaseTool):
    name = "write_topic_content"
    description = "Write or update markdown content for a topic at a given path. The content is written to the dot-prefixed topic file (e.g. .Music.md). Also writes to the underscore-prefixed antonym file if antonym_content is provided. Use this after generating content with your LLM to save it into the dialectics tree. Returns the file path written."
    parameters = {
        "type": "object",
        "properties": {
            "basepath": {"type": "string", "description": "Root path of the dialectics tree"},
            "path": {"type": "string", "description": "Path with / separator (e.g. '1/1/1')"},
            "content": {"type": "string", "description": "Markdown content to write. This replaces the entire file."},
            "antonym_content": {"type": "string", "description": "Optional content for the antonym file (_*.md)"},
            "create_subtopics": {
                "type": "boolean",
                "description": "If true and the path has 1/2/3 child dirs, create .Topic.md files for them too using their existing titles or generating from the path key.",
                "default": False,
            },
            "dry_run": {"type": "boolean", "description": "If true, only report what would be written", "default": True},
            "commit": {"type": "boolean", "description": "If false, write files but leave git changes unstaged. Only used when dry_run=False.", "default": True},
        },
        "required": ["basepath", "path", "content"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        basepath = kwargs["basepath"]
        path_key = kwargs["path"]
        content = kwargs["content"]
        antonym_content = kwargs.get("antonym_content")
        create_subtopics = kwargs.get("create_subtopics", False)
        dry_run = kwargs.get("dry_run", True)
        commit = kwargs.get("commit", True)

        dir_path = os.path.join(basepath, path_key)
        if not os.path.isdir(dir_path):
            return ToolResult.fail(self.name, f"Directory not found: {dir_path}", params=kwargs, duration_s=time.time() - t0)

        try:
            written = []

            # Write topic file
            topic_file = _find_topic_file(dir_path)
            if not topic_file:
                return ToolResult.fail(self.name, f"No topic file found in {dir_path}. Use rename_topic first to create one.", params=kwargs, duration_s=time.time() - t0)

            topic_path = os.path.join(dir_path, topic_file)
            written.append({
                "type": "topic_content",
                "file": topic_file,
                "path": path_key,
                "action": "write" if not dry_run else "dry_run",
                "char_count": len(content),
            })

            # Write antonym file
            antonym_written = None
            if antonym_content:
                antonym_file = _find_antonym_file(dir_path)
                if antonym_file:
                    ant_path = os.path.join(dir_path, antonym_file)
                    antonym_written = {
                        "type": "antonym_content",
                        "file": antonym_file,
                        "path": path_key,
                        "action": "write" if not dry_run else "dry_run",
                        "char_count": len(antonym_content),
                    }
                    written.append(antonym_written)

            # Check subtopics
            subtopics_found = []
            if create_subtopics:
                for key in ["1", "2", "3"]:
                    child_dir = os.path.join(dir_path, key)
                    if os.path.isdir(child_dir):
                        child_topic = _find_topic_file(child_dir)
                        if child_topic:
                            child_path = os.path.join(child_dir, child_topic)
                            subtopics_found.append({"key": key, "file": child_topic, "exists": True})
                        else:
                            subtopics_found.append({"key": key, "file": None, "exists": False})

            # Execute writes
            if not dry_run:
                with open(topic_path, "w") as f:
                    f.write(content)

                if antonym_content and antonym_written:
                    ant_path = os.path.join(dir_path, _find_antonym_file(dir_path))
                    with open(ant_path, "w") as f:
                        f.write(antonym_content)

                # Git commit
                if commit:
                    try:
                        from lib.os_tools import git_auto_commit
                        with git_auto_commit(basepath, f"write_content {path_key}: {topic_file}"):
                            pass
                    except Exception as e:
                        written.append({"type": "git_warning", "detail": str(e)})

            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "path": path_key,
                "topic_file": topic_file,
                "written": written,
                "file_count": len(written),
                "subtopics": subtopics_found if create_subtopics else [],
                "char_count": len(content),
                "dry_run": dry_run,
                "committed": commit and not dry_run,
            }, params=kwargs, duration_s=duration)

        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
