"""Parking stack for displaced topics. Like checkers: pop one off, slot another in.

Registry: JSON file at basepath/__park__.json
Each parked entry stores: original_path, topic_content, antonym_content, child_keys, timestamp.
"""
import json
import os
import re
import shutil
import time
from pathlib import Path

from .tool_base import BaseTool, ToolResult

TOPIC_FILE = re.compile(r"^\.[^.].*\.md$")
ANTONYM_FILE = re.compile(r"^_[^_].*\.md$")

PARK_FILE = "__park__.json"


def _load_park(basepath: str) -> list:
    p = os.path.join(basepath, PARK_FILE)
    if os.path.exists(p):
        return json.loads(Path(p).read_text("utf-8"))
    return []


def _save_park(basepath: str, park: list):
    Path(os.path.join(basepath, PARK_FILE)).write_text(
        json.dumps(park, indent=2, ensure_ascii=False), "utf-8"
    )


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


def _collect_child_keys(dir_path: str) -> list[str]:
    keys = []
    try:
        for name in sorted(os.listdir(dir_path)):
            if name in {"1", "2", "3"} and os.path.isdir(os.path.join(dir_path, name)):
                keys.append(name)
    except OSError:
        pass
    return keys


def _build_park_entry(basepath: str, path_key: str, parent_id: str = "") -> dict | None:
    """Build a park entry for a single path. Returns None if no topic file."""
    dir_path = os.path.join(basepath, path_key)
    if not os.path.isdir(dir_path):
        return None

    topic_file = _find_topic_file(dir_path)
    antonym_file = _find_antonym_file(dir_path)
    if not topic_file:
        return None

    topic_content = Path(os.path.join(dir_path, topic_file)).read_text("utf-8")
    antonym_content = Path(os.path.join(dir_path, antonym_file)).read_text("utf-8") if antonym_file else ""
    child_keys = _collect_child_keys(dir_path)

    return {
        "id": f"parked_{path_key.replace('/', '_')}",
        "original_path": path_key,
        "topic_title": topic_file[1:-3] if topic_file else "",
        "topic_file": topic_file,
        "antonym_file": antonym_file or "",
        "topic_content": topic_content,
        "antonym_content": antonym_content,
        "child_keys": child_keys,
        "has_children": len(child_keys) > 0,
        "parent_id": parent_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


class ParkTopicTool(BaseTool):
    name = "park_topic"
    description = "Park a topic on the displacement stack. Reads topic content, antonym, and child structure from a path, stores them in __park__.json, and clears the directory. Use when you need to vacate a position for another topic. With recursive=True, also parks all child triples (1/2/3) as separate entries."
    parameters = {
        "type": "object",
        "properties": {
            "basepath": {"type": "string", "description": "Root path of the dialectics tree"},
            "path": {"type": "string", "description": "Path with / separator (e.g. '2/1/3')"},
            "recursive": {"type": "boolean", "description": "If true, also park all child triples recursively", "default": False},
            "keep_structure": {"type": "boolean", "description": "If true, keep dirs but clear .md files. If false, delete dirs.", "default": False},
            "dry_run": {"type": "boolean", "description": "If true, only report what would be parked", "default": True},
        },
        "required": ["basepath", "path"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        basepath = kwargs["basepath"]
        path_key = kwargs["path"]
        recursive = kwargs.get("recursive", False)
        keep_structure = kwargs.get("keep_structure", False)
        dry_run = kwargs.get("dry_run", True)

        dir_path = os.path.join(basepath, path_key)
        if not os.path.isdir(dir_path):
            return ToolResult.fail(self.name, f"Directory not found: {dir_path}", params=kwargs, duration_s=time.time() - t0)

        try:
            # Collect all entries to park (self + children if recursive)
            entries = []
            entry = _build_park_entry(basepath, path_key, parent_id="")
            if entry:
                entries.append(entry)

            if recursive:
                for child_key in _collect_child_keys(dir_path):
                    child_path = f"{path_key}/{child_key}"
                    child_entry = _build_park_entry(basepath, child_path, parent_id=entry["id"] if entry else "")
                    if child_entry:
                        entries.append(child_entry)
                        # Grandchildren too
                        for grand_key in _collect_child_keys(os.path.join(dir_path, child_key)):
                            grand_path = f"{child_path}/{grand_key}"
                            grand_entry = _build_park_entry(basepath, grand_path, parent_id=child_entry["id"])
                            if grand_entry:
                                entries.append(grand_entry)

            parked_ids = [e["id"] for e in entries if e["id"]]

            if not dry_run:
                park = _load_park(basepath)
                park.extend(entries)
                _save_park(basepath, park)

                # Clear filesystem
                for e in entries:
                    ep = os.path.join(basepath, e["original_path"])
                    if keep_structure and os.path.isdir(ep):
                        for fname in os.listdir(ep):
                            fpath = os.path.join(ep, fname)
                            if os.path.isfile(fpath) and fname.endswith(".md"):
                                os.remove(fpath)
                    elif os.path.isdir(ep):
                        shutil.rmtree(ep)

            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "parked_entries": entries,
                "entry_count": len(entries),
                "park_ids": parked_ids,
                "topic_title": entry["topic_title"] if entry else "",
                "has_children": entry["has_children"] if entry else False,
                "child_keys": entry["child_keys"] if entry else [],
                "recursive": recursive,
                "dry_run": dry_run,
            }, params=kwargs, duration_s=duration)

        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)


class InsertTopicTool(BaseTool):
    name = "insert_topic"
    description = "Insert a parked topic at a target path. If the target already has content, that topic gets parked first (checkers displacement). With restore_children=True, also restores child triples from the park stack that have matching parent_id."
    parameters = {
        "type": "object",
        "properties": {
            "basepath": {"type": "string", "description": "Root path of the dialectics tree"},
            "target_path": {"type": "string", "description": "Where to insert the parked topic"},
            "park_id": {"type": "string", "description": "ID of the parked topic"},
            "restore_children": {"type": "boolean", "description": "If true, also restore child triples", "default": False},
            "dry_run": {"type": "boolean", "description": "If true, only report what would happen", "default": True},
        },
        "required": ["basepath", "target_path", "park_id"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        basepath = kwargs["basepath"]
        target_path = kwargs["target_path"]
        park_id = kwargs["park_id"]
        restore_children = kwargs.get("restore_children", False)
        dry_run = kwargs.get("dry_run", True)

        target_dir = os.path.join(basepath, target_path)

        try:
            park = _load_park(basepath)
            entry = None
            park_idx = None
            for i, e in enumerate(park):
                if e["id"] == park_id:
                    entry = e
                    park_idx = i
                    break

            if not entry:
                return ToolResult.fail(self.name, f"Parked topic '{park_id}' not found in {PARK_FILE}", params=kwargs, duration_s=time.time() - t0)

            actions = []

            # Also find children to restore (entries with matching parent_id)
            child_entries = []
            if restore_children and entry.get("child_keys"):
                for child_key in entry["child_keys"]:
                    expected_child_path = f"{entry['original_path']}/{child_key}"
                    for ce in park:
                        if ce.get("original_path") == expected_child_path:
                            child_entries.append(ce)

            # Check if target already has content -> displace
            displaced = None
            if os.path.isdir(target_dir):
                existing_topic = _find_topic_file(target_dir)
                if existing_topic:
                    existing_content = Path(os.path.join(target_dir, existing_topic)).read_text("utf-8")
                    existing_ant = _find_antonym_file(target_dir)
                    existing_ant_content = Path(os.path.join(target_dir, existing_ant)).read_text("utf-8") if existing_ant else ""
                    displaced = {
                        "original_path": target_path,
                        "topic_title": existing_topic[1:-3],
                        "topic_content": existing_content,
                        "antonym_content": existing_ant_content,
                    }
                    actions.append({"type": "displace", "detail": f"Parking existing '{displaced['topic_title']}' at {target_path}"})

            actions.append({"type": "write_topic", "file": entry["topic_file"], "content_len": len(entry["topic_content"])})
            if entry["antonym_content"]:
                actions.append({"type": "write_antonym", "file": entry["antonym_file"], "content_len": len(entry["antonym_content"])})

            child_actions = []
            if child_entries:
                for ce in child_entries:
                    child_target = f"{target_path}/{ce['original_path'].split('/')[-1]}"
                    child_actions.append({
                        "child_path": child_target,
                        "title": ce["topic_title"],
                        "file": ce["topic_file"],
                    })
                actions.append({"type": "restore_children", "count": len(child_entries), "children": child_actions})

            if not dry_run:
                os.makedirs(target_dir, exist_ok=True)

                # Displace existing content
                if displaced:
                    displaced_entry = {
                        "id": f"parked_{int(t0)}",
                        "original_path": target_path,
                        "topic_title": displaced["topic_title"],
                        "topic_file": f".{displaced['topic_title']}.md",
                        "antonym_file": f"_{displaced['topic_title']}.md",
                        "topic_content": displaced["topic_content"],
                        "antonym_content": displaced["antonym_content"],
                        "child_keys": [],
                        "has_children": False,
                        "parent_id": "",
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    park.append(displaced_entry)
                    for fname in os.listdir(target_dir):
                        fpath = os.path.join(target_dir, fname)
                        if os.path.isfile(fpath) and fname.endswith(".md"):
                            os.remove(fpath)

                # Write main topic
                Path(os.path.join(target_dir, entry["topic_file"])).write_text(entry["topic_content"], "utf-8")
                if entry["antonym_content"]:
                    Path(os.path.join(target_dir, entry["antonym_file"])).write_text(entry["antonym_content"], "utf-8")

                # Restore children
                for ce in child_entries:
                    child_target_dir = os.path.join(target_dir, ce['original_path'].split('/')[-1])
                    os.makedirs(child_target_dir, exist_ok=True)
                    Path(os.path.join(child_target_dir, ce["topic_file"])).write_text(ce["topic_content"], "utf-8")
                    if ce["antonym_content"]:
                        Path(os.path.join(child_target_dir, ce["antonym_file"])).write_text(ce["antonym_content"], "utf-8")

                # Remove consumed entries from park
                consumed_ids = {entry["id"]}
                if child_entries:
                    consumed_ids.update(ce["id"] for ce in child_entries)
                park = [e for e in park if e["id"] not in consumed_ids]
                _save_park(basepath, park)

            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "park_id": park_id,
                "topic_title": entry["topic_title"],
                "target_path": target_path,
                "actions": actions,
                "action_count": len(actions),
                "displaced": displaced is not None,
                "displaced_title": displaced["topic_title"] if displaced else None,
                "children_restored": len(child_entries),
                "dry_run": dry_run,
            }, params=kwargs, duration_s=duration)

        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)


class ListParkedTool(BaseTool):
    name = "list_parked"
    description = "List all currently parked topics on the displacement stack. Shows each with ID, original path, title, parent_id, and children."
    parameters = {
        "type": "object",
        "properties": {
            "basepath": {"type": "string", "description": "Root path of the dialectics tree"},
        },
        "required": ["basepath"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        basepath = kwargs["basepath"]
        try:
            park = _load_park(basepath)
            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "parked_count": len(park),
                "parked": [
                    {
                        "id": e["id"],
                        "original_path": e["original_path"],
                        "topic_title": e["topic_title"],
                        "parent_id": e.get("parent_id", ""),
                        "has_children": e.get("has_children", False),
                        "child_keys": e.get("child_keys", []),
                        "timestamp": e.get("timestamp", ""),
                    }
                    for e in park
                ],
            }, params=kwargs, duration_s=duration)
        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
