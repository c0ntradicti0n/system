import os
from pprint import pprint

import regex
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
def get_filename_without_extension(path):
    return os.path.splitext(os.path.basename(path))[0]

document_store = InMemoryDocumentStore(use_bm25=True)


doc_dir = os.environ["SYSTEM"]

# 2. Index the documents and store the file paths as metadata:
all_files = []
for root, _, files in os.walk(doc_dir):
    for file in files:
        all_files.append(os.path.join(root, file))

documents = []
for file_path in all_files:
    if not file_path.endswith(".md"):
        continue
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        topic = get_filename_without_extension(file_path)
        documents.append(
            {
                "content": topic +  "\n\n" + text,
                "meta": {"file_path": file_path},  # Storing file path as metadata
            }
        )

document_store.write_documents(documents)

retriever = BM25Retriever(document_store=document_store)

# Use DistilBERT for question-answering
reader = FARMReader(model_name_or_path="distilbert-base-cased-distilled-squad", use_gpu=True)

pipe = ExtractiveQAPipeline(reader, retriever)


def main(string):
    prediction = pipe.run(
        query=string,
        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}},
    )
    print(prediction)
    return [
        {
            "answer": answer,
            "path": next(
                regex.finditer(
                    "(\/[1-3\._])+", answer.meta["file_path"].replace(doc_dir, "")
                ),
                "",
            )
            .group()
            .strip("/"),
        }
        for answer in prediction["answers"]
    ]
