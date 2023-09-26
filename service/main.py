import os
from pprint import pprint

import regex
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from lib.doc import get_documents
from lib.t import catchtime

with catchtime("init"):
    persist_directory = ".chroma"
    embedding = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
    )
    collection_name = "system"

    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        doc_dir = os.environ["SYSTEM"]
        documents = get_documents(doc_dir)

        vector_store = Chroma.from_documents(
            embedding=embedding,
            documents=documents,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        vector_store.persist()
    else:
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding,
        )


def search(query, top_k=5, filter_path=""):
    if filter_path:
        filter = {filter_path: {"$eq": filter_path}}
    else:
        filter = None

    results = vector_store.search(query, k=top_k, search_type="mmr", filter=filter)
    return results


if __name__ == "__main__":
    with catchtime("search"):
        pprint(search("sex", top_k=4))
    with catchtime("search"):
        pprint(search("heart", top_k=4))
    with catchtime("search"):
        pprint(search("mass", top_k=4))
    with catchtime("search"):
        pprint(search("how does the system start?", top_k=4))
    with catchtime("search"):
        pprint(search("best food?", top_k=4))

    with catchtime("filter"):
        pprint(search("best food?", top_k=4, filter_path="2"))

    with catchtime("search"):
        pprint(search("best food?", filter_path="2"))
