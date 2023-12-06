import os

from langchain.text_splitter import CharacterTextSplitter
from regex import regex


def get_filename_without_extension(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_documents(document_dir):
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)

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
        except:
            raise ValueError(f"Error parsing path for {file_path}")
        clean_path = path.replace("/", "")
        docs = text_splitter.create_documents(
            [content],
            metadatas=[
                {
                    "file_path": file_path.replace(document_dir, ""),
                    "path": path,
                    **{
                        clean_path[:i]: clean_path[:i]
                        for i in range(clean_path.__len__())
                    },
                }
            ],
        )
        documents.extend(docs)
    return documents
