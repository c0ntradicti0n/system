import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from lib import config
from lib.helper import tree

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
)
model = os.environ.get("LLM_MODEL", "deepseek-chat")


def walk_md_files(root):
    """Walk dir tree, yield .md files (skip .de.md, .git, .idea, states)."""
    skip_dirs = {".git", ".idea", "states", "__pycache__"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for f in sorted(filenames):
            if f.endswith(".de.md"):
                continue
            if not f.endswith(".md"):
                continue
            if f.startswith(".") or f.startswith("_"):
                yield Path(dirpath) / f


def has_de_translation(fp):
    """Check if any .de.md with same prefix already exists in fp's dir."""
    prefix = fp.name[0]  # '.' or '_'
    for other in fp.parent.iterdir():
        if other.name.endswith('.de.md') and other.name.startswith(prefix):
            return True
    return False


def translate_batch(client, batch, target_lang, model, root):
    """Send batch of files to LLM, return {filename: translation}.
    Keys are translated filenames (e.g. .Ontologie.md), values are content."""
    prompt_parts = []
    for fp in batch:
        rel = fp.relative_to(root)
        prompt_parts.append(f"=== {rel} ===\n{fp.read_text('utf-8')}")

    system = (
        f"Translate each file to {target_lang}. Output ONE JSON key per file.\n"
        "Rules:\n"
        "- TRANSLATE THE FILENAME itself (the concept name after . or _) to German\n"
        "- Keep the same prefix (. or _) as the original\n"
        "- The JSON must have exactly one key per input file\n"
        "- Each key is the translated relative path (e.g. '1/3/1/1/.Physik.md')\n"
        "- Each value is the translated Markdown content\n"
        "- Preserve all wiki-links, Markdown formatting, blank lines, structure\n"
        "- Do not add extra keys or commentary outside the JSON"
    )

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(prompt_parts)},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:500]}")

    if len(data) != len(batch):
        raise ValueError(f"Expected {len(batch)} keys, got {len(data)}: {list(data.keys())}")

    return data


def write_translations(base_path, translations):
    """Write translated content as .de.md files (basename translated by LLM)."""
    for rel_path, content in translations.items():
        p = Path(base_path) / rel_path
        de_path = p.with_name(p.stem + '.de.md')
        de_path.parent.mkdir(parents=True, exist_ok=True)
        # remove old .de.md with English stem if it exists
        # (the English-stem file is now superseded by the German-stem one)
        de_path.write_text(content, encoding="utf-8")
        print(f"  wrote {de_path}")


def main():
    parser = argparse.ArgumentParser(description="Translate dialectics tree files via LLM")
    parser.add_argument("path", nargs="?", default="",
                        help="Relative path under SYSTEM dir (e.g. 1/3/1/1)")
    parser.add_argument("--lang", default="de", help="Target language code")
    parser.add_argument("--dry-run", action="store_true", help="List files without translating")
    parser.add_argument("--batch-size", type=int, default=10, help="Files per API call")
    args = parser.parse_args()

    root = os.path.join(config.system_path, args.path)
    if not os.path.isdir(root):
        print(f"Error: {root} is not a directory")
        return 1

    all_files = list(walk_md_files(root))
    if not all_files:
        print(f"No .md files found under {root}")
        return 0

    print(f"Translating: {root}")
    print(f"Language:    {args.lang}")
    print(f"Model:       {model}")
    print(f"Files:       {len(all_files)}")

    # filter: skip directories that already have a .de.md translation
    todo = []
    for fp in all_files:
        if has_de_translation(fp):
            print(f"  skip (exists): {fp}")
        else:
            todo.append(fp)

    if not todo:
        print("All files already translated.")
        return 0

    print(f"To translate: {len(todo)} files")

    if args.dry_run:
        print("\nDry-run — files that would be translated (German filename TBD by LLM):")
        for fp in todo:
            print(f"  {fp}")
        return 0

    # batch and translate
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    for i, batch in enumerate(batches):
        print(f"\nBatch {i + 1}/{len(batches)} ({len(batch)} files)...")
        try:
            result = translate_batch(client, batch, args.lang, model, root)
            write_translations(root, result)
            print(f"  batch {i + 1} done")
        except Exception as e:
            print(f"  ERROR: {e}")
            print(f"  Skipping batch {i + 1}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    exit(main())
