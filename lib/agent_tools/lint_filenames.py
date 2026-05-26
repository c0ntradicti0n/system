import os
import re
import time

from .tool_base import BaseTool, ToolResult

# Valid single-char triple keys — these are structural, not content names
TRIPLE_KEYS = {"1", "2", "3", ".", "_"}
# System dirs/files to skip entirely
SYSTEM_NAMES = {".git", ".gitignore", ".gitkeep", ".idea", ".DS_Store", "__pycache__",
                "node_modules", ".certs", ".cache", ".caches", ".helm-chart"}
# System prefixes (dotfiles at root)
SYSTEM_PREFIXES = (".git", ".idea", ".DS_Store", "__pycache__")
# Characters that are problematic in content titles
BAD_CHARS = re.compile(r"""['"”:;!@#$%^&*()+={}\[\]|\\,<>?~`]""")
# Valid title: starts with key prefix, then space/dot, then alphanumeric title
KEY_PREFIX = re.compile(r"^([1-3._])([ .])(.+)")


def should_skip(name: str) -> bool:
    """Skip system files and valid structural entries."""
    if name in TRIPLE_KEYS:
        return True
    if name in SYSTEM_NAMES:
        return True
    if name.startswith(SYSTEM_PREFIXES):
        return True
    # .md files that are just hidden files (like .System.md) at root — allow
    if name.startswith(".") and name.endswith(".md") and len(name) > 4:
        return True
    return False


def lint_filepath(root: str, rel_path: str, dry_run: bool) -> dict | None:
    """Check a single file/directory path for naming violations."""
    full_path = os.path.join(root, rel_path)
    name = os.path.basename(rel_path)

    if should_skip(name):
        return None

    violations = []

    # Try to parse as key-prefixed content name
    m = KEY_PREFIX.match(name)
    if m:
        _prefix, _sep, title = m.groups()
        # Remove .md extension for title check
        title_clean = title[:-3] if title.endswith(".md") else title
        bad = BAD_CHARS.findall(title_clean)
        if bad:
            violations.append({
                "type": "bad_characters",
                "chars_found": sorted(set(bad)),
                "in_part": "title",
                "suggested_fix": f"{_prefix}{_sep}{BAD_CHARS.sub('', title_clean)}{'.md' if title.endswith('.md') else ''}",
            })
    else:
        # Not a key-prefixed name — check whole name for bad chars
        bad = BAD_CHARS.findall(name)
        if bad:
            violations.append({
                "type": "bad_characters",
                "chars_found": sorted(set(bad)),
                "in_part": "filename",
                "suggested_fix": BAD_CHARS.sub("", name),
            })

        # Files at root level that aren't key-prefixed but are .md — flag as unconventional
        if name.endswith(".md") and "/" not in rel_path:
            violations.append({
                "type": "unconventional_name",
                "detail": f"Root-level .md file without key prefix: {name}",
            })

    if not violations:
        return None

    result = {
        "path": rel_path,
        "filename": name,
        "violations": violations,
    }

    if not dry_run and violations:
        fixed_name = BAD_CHARS.sub("", name)
        if fixed_name and fixed_name != name:
            old_full = full_path
            parent = os.path.dirname(full_path)
            new_full = os.path.join(parent, fixed_name)
            try:
                os.rename(old_full, new_full)
                result["renamed_to"] = fixed_name
            except OSError as e:
                result["rename_error"] = str(e)

    return result


class LintFilenamesTool(BaseTool):
    name = "lint_filenames"
    description = "Scan and optionally fix filenames in the dialectical filesystem. Catches bad chars like quotes, colons, exclamation marks. Can autofix with dry_run=False."
    parameters = {
        "type": "object",
        "properties": {
            "basepath": {"type": "string", "description": "Root path to scan"},
            "dry_run": {"type": "boolean", "description": "If true, only report violations without renaming", "default": True},
            "pattern": {"type": "string", "description": "Regex pattern for valid filenames (default: alphanumeric + spaces + hyphens + dots)"},
        },
        "required": ["basepath"],
    }

    def __call__(self, **kwargs) -> ToolResult:
        t0 = time.time()
        basepath = kwargs["basepath"]
        dry_run = kwargs.get("dry_run", True)

        if not os.path.isdir(basepath):
            return ToolResult.fail(self.name, f"Path not found: {basepath}", params=kwargs, duration_s=time.time() - t0)

        all_violations = []
        fixed_count = 0

        try:
            for dirpath, dirnames, filenames in os.walk(basepath):
                rel_dir = os.path.relpath(dirpath, basepath)
                if rel_dir == ".":
                    rel_dir = ""

                # Check all files and dirs
                for name in filenames:
                    rel_path = os.path.join(rel_dir, name) if rel_dir else name
                    result = lint_filepath(basepath, rel_path, dry_run)
                    if result:
                        if "renamed_to" in result:
                            fixed_count += 1
                        all_violations.append(result)

                # Check directory names too
                for name in dirnames:
                    rel_path = os.path.join(rel_dir, name) if rel_dir else name
                    result = lint_filepath(basepath, rel_path, dry_run)
                    if result:
                        if "renamed_to" in result:
                            fixed_count += 1
                        all_violations.append(result)

            duration = time.time() - t0
            return ToolResult.ok(self.name, {
                "violations": all_violations,
                "violation_count": len(all_violations),
                "fixed_count": fixed_count,
                "dry_run": dry_run,
            }, params=kwargs, duration_s=duration)

        except Exception as e:
            return ToolResult.fail(self.name, str(e), params=kwargs, duration_s=time.time() - t0)
