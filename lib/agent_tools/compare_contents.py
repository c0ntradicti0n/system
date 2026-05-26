import os
import time
from pathlib import Path

from ._path import _project_root
from .tool_base import BaseTool, ToolResult


def _read_content(basepath: str, path_key: str) -> str | None:
    """Read first .md content file under basepath/path_key/. Returns None if not found."""
    search_dir = os.path.join(basepath, path_key)
    if not os.path.isdir(search_dir):
        return None
    for fname in sorted(os.listdir(search_dir)):
        if fname.endswith(".md"):
            try:
                return Path(os.path.join(search_dir, fname)).read_text("utf-8")[:2000]
            except Exception:
                pass
    return None


def _check_triple_structure(basepath: str, path_key: str) -> list:
    """Check if a path has the expected triple structure (1/2/3 dirs, .topic, _antonym)."""
    issues = []
    dir_path = os.path.join(basepath, path_key)
    if not os.path.isdir(dir_path):
        return [{"type": "missing_directory", "path": path_key}]

    names = set(os.listdir(dir_path))
    has_children = {"1", "2", "3"}.issubset(
        {n for n in names if os.path.isdir(os.path.join(dir_path, n))}
    )
    has_topic = any(n.startswith(".") and n.endswith(".md") for n in names)
    has_antonym = any(n.startswith("_") and n.endswith(".md") for n in names)

    if not has_children:
        issues.append({"type": "incomplete_triple", "detail": f"Missing 1/2/3 subdirectories", "path": path_key})
    if not has_topic:
        issues.append({"type": "missing_topic", "detail": "No .md topic file found", "path": path_key})
    if not has_antonym:
        issues.append({"type": "missing_antonym", "detail": "No _antonym.md file found", "path": path_key})
    return issues


class CompareContentsTool(BaseTool):
    name = "compare_contents"
    description = "Compare content across two paths in the fractal tree. Checks structural symmetry, content similarity, and antonym opposition."
    parameters = {
        "type": "object",
        "properties": {
            "basepath": {"type": "string", "description": "Filesystem base path for content files"},
            "path_a": {"type": "string", "description": "First path with / separator (e.g. '1/3/1/1')"},
            "path_b": {"type": "string", "description": "Second path with / separator (e.g. '1/3/1/2')"},
            "mode": {
                "type": "string",
                "enum": ["fuzzy", "semantic"],
                "description": "Comparison mode: fuzzy (string similarity) or semantic (vector embedding)",
                "default": "fuzzy",
            },
            "threshold": {"type": "number", "description": "Similarity threshold for flagging", "default": 0.5},
        },
        "required": ["basepath", "path_a", "path_b"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        basepath = kwargs["basepath"]
        path_a = kwargs["path_a"]
        path_b = kwargs["path_b"]
        mode = kwargs.get("mode", "fuzzy")
        threshold = kwargs.get("threshold", 0.5)

        inconsistencies = []

        # 1. Structural completeness check (missing entries, not inconsistencies)
        struct_a = _check_triple_structure(basepath, path_a)
        struct_b = _check_triple_structure(basepath, path_b)
        completeness_issues = struct_a + struct_b

        # 2. Read content
        content_a = _read_content(basepath, path_a)
        content_b = _read_content(basepath, path_b)

        if content_a is None:
            inconsistencies.append({"type": "no_content", "path": path_a})
        if content_b is None:
            inconsistencies.append({"type": "no_content", "path": path_b})

        similarity_score = 0.0

        if content_a and content_b:
            if mode == "fuzzy":
                try:
                    from philo.compare import weighted_fuzzy_compare
                    similar, score = weighted_fuzzy_compare(
                        content_a[:200], content_b[:200], threshold
                    )
                    similarity_score = score
                    if score > threshold * 1.5:
                        inconsistencies.append({
                            "type": "too_similar",
                            "detail": f"Thesis and antithesis content are too similar (score={score:.2f}) — may not be genuine opposites",
                            "similarity_score": round(score, 3),
                        })
                except Exception as e:
                    inconsistencies.append({"type": "compare_error", "detail": str(e)})

            elif mode == "semantic":
                try:
                    from lib.vector_store import get_embedding_function
                    ef = get_embedding_function()
                    emb_a = ef.embed_documents([content_a[:500]])[0]
                    emb_b = ef.embed_documents([content_b[:500]])[0]
                    # cosine similarity
                    dot = sum(x * y for x, y in zip(emb_a, emb_b))
                    norm_a = sum(x * x for x in emb_a) ** 0.5
                    norm_b = sum(x * x for x in emb_b) ** 0.5
                    if norm_a and norm_b:
                        similarity_score = dot / (norm_a * norm_b)
                        if similarity_score > 0.85:
                            inconsistencies.append({
                                "type": "too_similar_semantic",
                                "detail": f"Semantic similarity too high ({similarity_score:.3f}) — not genuine opposites",
                                "similarity_score": round(similarity_score, 3),
                            })
                except Exception as e:
                    inconsistencies.append({"type": "semantic_error", "detail": str(e)})

        # 3. Content length imbalance
        if content_a and content_b:
            len_a, len_b = len(content_a), len(content_b)
            if min(len_a, len_b) > 0 and max(len_a, len_b) / min(len_a, len_b) > 3:
                inconsistencies.append({
                    "type": "length_imbalance",
                    "detail": f"Content length mismatch: {len_a} vs {len_b} chars",
                    "length_a": len_a,
                    "length_b": len_b,
                })

        duration = time.time() - t0
        return ToolResult.ok(self.name, {
            "path_a": path_a,
            "path_b": path_b,
            "content_a_preview": (content_a or "")[:200],
            "content_b_preview": (content_b or "")[:200],
            "similarity_score": round(similarity_score, 3),
            "completeness_issues": completeness_issues,
            "inconsistencies": inconsistencies,
            "inconsistency_count": len(inconsistencies),
        }, params=kwargs, duration_s=duration)
