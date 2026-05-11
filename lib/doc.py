import os
from regex import regex


def get_filename_without_extension(path):
    return os.path.splitext(os.path.basename(path))[0]


def _chunk_text(text, chunk_size=3000):
    """Split text into non-overlapping chunks of at most chunk_size chars."""
    chunks = []
    while len(text) > chunk_size:
        split_at = text.rfind(' ', 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def get_documents(document_dir):
    all_files = []
    for root, _, files in os.walk(document_dir):
        for file in files:
            all_files.append(os.path.join(root, file))

    all_files = [x for x in all_files if ".git" not in x]

    documents = []
    for file_path in all_files:
        if not file_path.endswith(".md"):
            continue
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        topic = get_filename_without_extension(file_path)
        content = topic + "\n\n" + text
        try:
            path = regex.match(
                r"^[\/1-3_]*", file_path.replace(document_dir, "")
            ).group(0)
        except Exception:
            raise ValueError(f"Error parsing path for {file_path}")
        clean_path = path.replace("/", "")
        metadata = {
            "file_path": file_path.replace(document_dir, ""),
            "path": path,
            **{clean_path[:i]: clean_path[:i] for i in range(len(clean_path))},
        }
        for chunk in _chunk_text(content):
            documents.append({"page_content": chunk, "metadata": metadata})
    return documents
