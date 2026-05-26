import os
import re
import time
from pathlib import Path

from .tool_base import BaseTool, ToolResult


def _get_tree_path(abspath: str, basepath: str) -> str:
    """Convert absolute path to tree-relative path (e.g. '3/2/1/3/2/1')."""
    rel = os.path.relpath(abspath, basepath)
    parts = rel.split(os.sep)
    # Filter out the leading parts before we hit the triple key pattern
    tree_parts = [p for p in parts if re.match(r"^[1-3._]$", p)]
    return "/".join(tree_parts)


def _get_parent_context(abspath: str, basepath: str) -> dict:
    """Get the parent directory's topic title and structure context."""
    parent = os.path.dirname(abspath)
    if not parent or parent == basepath:
        return {"parent_path": "", "parent_topic": "(root)", "siblings": []}

    rel = os.path.relpath(parent, basepath)
    tree_path = _get_tree_path(parent, basepath)

    # Find topic file in parent
    topic = ""
    siblings = []
    try:
        for f in sorted(os.listdir(parent)):
            if f.startswith(".") and f.endswith(".md"):
                topic = f[1:-3]  # strip . prefix and .md suffix
            elif os.path.isdir(os.path.join(parent, f)):
                dir_topic = ""
                for sf in os.listdir(os.path.join(parent, f)):
                    if sf.startswith(".") and sf.endswith(".md"):
                        dir_topic = sf[1:-3]
                        break
                siblings.append({"key": f, "topic": dir_topic})
    except OSError:
        pass

    return {
        "parent_path": tree_path,
        "parent_topic": topic,
        "siblings": siblings,
    }


def _find_concept_in_filesystem(basepath: str, concept: str, content_search: bool = False) -> list:
    """Find all occurrences of concept in the dialectics filesystem."""
    results = []
    concept_lower = concept.lower()

    for dirpath, dirnames, filenames in os.walk(basepath):
        # Skip system dirs
        rel = os.path.relpath(dirpath, basepath)
        if rel.startswith("."):
            continue

        # Check filenames for concept match
        for fname in filenames:
            if not fname.endswith(".md"):
                continue
            # Get title (strip . or _ prefix and .md suffix)
            title = fname[1:-3] if fname.startswith((".", "_")) else fname[:-3]
            match_type = None
            match_detail = ""

            # Exact or fuzzy title match
            if title.lower() == concept_lower:
                match_type = "exact_title"
                match_detail = f"Exact title match: {title}"
            elif concept_lower in title.lower():
                match_type = "partial_title"
                match_detail = f"Partial title match: {title}"
            elif content_search:
                fpath = os.path.join(dirpath, fname)
                try:
                    content = Path(fpath).read_text("utf-8")
                    if concept_lower in content.lower():
                        match_type = "content"
                        # Find context around first mention
                        idx = content.lower().find(concept_lower)
                        start = max(0, idx - 40)
                        end = min(len(content), idx + len(concept) + 40)
                        match_detail = f"Content mention: ...{content[start:end].strip()}..."
                except Exception:
                    pass

            if match_type:
                fpath = os.path.join(dirpath, fname)
                tree_path = _get_tree_path(fpath, basepath)
                context = _get_parent_context(fpath, basepath)

                # Get full triple context (keys on the path)
                triple_keys = [p for p in os.path.relpath(fpath, basepath).split(os.sep)
                               if re.match(r"^[1-3._]$", p)]

                results.append({
                    "path": os.path.relpath(fpath, basepath),
                    "tree_path": tree_path,
                    "triple_keys": "/".join(triple_keys),
                    "depth": len(triple_keys),
                    "match_type": match_type,
                    "match_detail": match_detail,
                    "topic_title": title,
                    "context": context,
                })

    return results


class SearchConceptTool(BaseTool):
    name = "search_concept"
    description = "Search for a concept across the dialectical tree. Finds all files with matching titles, partial titles, or content mentions. Returns each occurrence with its tree position, parent context, and depth. Use this to find where concepts are scattered across unrelated branches."
    parameters = {
        "type": "object",
        "properties": {
            "concept": {"type": "string", "description": "Concept to search for (e.g. 'music', 'being', 'time')"},
            "basepath": {"type": "string", "description": "Root path of the dialectics tree"},
            "content_search": {"type": "boolean", "description": "Also search within file contents (slower)", "default": False},
            "group_by_path": {"type": "boolean", "description": "Group results by tree path prefix to show clustering", "default": True},
        },
        "required": ["concept", "basepath"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        concept = kwargs["concept"]
        basepath = kwargs["basepath"]
        content_search = kwargs.get("content_search", False)
        group_by_path = kwargs.get("group_by_path", True)

        if not os.path.isdir(basepath):
            return ToolResult.fail(self.name, f"Path not found: {basepath}", params=kwargs, duration_s=time.time() - t0)

        try:
            results = _find_concept_in_filesystem(basepath, concept, content_search=content_search)

            # Group results by first segment of tree path (topic root)
            groups = {}
            if group_by_path:
                for r in results:
                    tree = r.get("tree_path", "")
                    root = tree.split("/")[0] if tree else "root"
                    prefix_parts = tree.split("/")[:3] if tree else []
                    prefix = "/".join(prefix_parts) if prefix_parts else "root"
                    groups.setdefault(prefix, []).append(r)

            duration = time.time() - t0

            result_data = {
                "concept": concept,
                "total_occurrences": len(results),
                "exact_title_matches": len([r for r in results if r["match_type"] == "exact_title"]),
                "partial_title_matches": len([r for r in results if r["match_type"] == "partial_title"]),
                "content_mentions": len([r for r in results if r["match_type"] == "content"]),
                "results": results,
                "grouped": groups if group_by_path else {},
                "group_count": len(groups) if group_by_path else 0,
            }

            # Add analysis
            if group_by_path and len(groups) > 1:
                result_data["analysis"] = (
                    f"Concept '{concept}' appears in {len(groups)} different tree branches: "
                    + ", ".join(sorted(groups.keys()))
                    + ". This suggests the concept is scattered across unrelated positions "
                    "rather than uniquely located."
                )
            elif len(results) > 0:
                result_data["analysis"] = (
                    f"Concept '{concept}' found in {len(results)} location(s), "
                    f"all within a single tree branch."
                )
            else:
                result_data["analysis"] = f"Concept '{concept}' not found in the tree."

            return ToolResult.ok(self.name, result_data, params=kwargs, duration_s=duration)

        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
