import os

import regex
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import (BM25Retriever, DensePassageRetriever,
                            EmbeddingRetriever, FARMReader)
from haystack.pipelines import (DocumentSearchPipeline, ExtractiveQAPipeline,
                                Pipeline)


def get_filename_without_extension(path):
    return os.path.splitext(os.path.basename(path))[0]


document_store = InMemoryDocumentStore(use_bm25=True, bm25_algorithm="BM25Plus")

doc_dir = os.environ["SYSTEM"]

# 2. Index the documents and store the file paths as metadata:
all_files = []
for root, _, files in os.walk(doc_dir):
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
        documents.append(
            {
                "content": (topic + "\n\n" + text)
                .replace(".md", "")
                .replace("_", "")
                .lower(),
                "meta": {"file_path": file_path},  # Storing file path as metadata
            }
        )
document_store.write_documents(documents)

from haystack.nodes import JoinDocuments

# Initialize Sparse Retriever
bm25_retriever = BM25Retriever(document_store=document_store)

# Initialize Reader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Create ensembled pipeline
p_ensemble = Pipeline()
p_ensemble.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
p_ensemble.add_node(
    component=JoinDocuments(join_mode="concatenate"),
    name="JoinResults",
    inputs=["BM25Retriever"],
)
p_ensemble.add_node(component=reader, name="Reader", inputs=["JoinResults"])

# Uncomment the following to generate the pipeline image
# p_ensemble.draw("pipeline_ensemble.png")

# Run pipeline
pipe = p_ensemble


def main(string):
    prediction = pipe.run(
        query=string,
        params={"BM25Retriever": {"top_k": 5}},
    )
    print(f"{string=},{prediction=}")
    return [
        {
            "content": d.content[:100],
            "answer": d.anwswer if hasattr(d, "answer") else None,
            "path": next(
                regex.finditer(
                    "(\/[1-3\._])+", d.meta["file_path"].replace(doc_dir, "")
                ),
                "",
            )
            .group()
            .strip("/"),
        }
        for d in prediction["documents"]
    ]
