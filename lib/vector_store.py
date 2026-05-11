import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from lib.doc import get_documents

_MODEL = "sentence-transformers/all-mpnet-base-v2"


class VectorStore:
    def __init__(self, collection):
        self.collection = collection

    def search(self, query, k=5, search_type=None, filter=None):
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=filter or None,
        )
        return [
            {"page_content": doc, "metadata": meta}
            for doc, meta in zip(
                results["documents"][0], results["metadatas"][0]
            )
        ]


def get_vector_store(persist_directory):
    ef = SentenceTransformerEmbeddingFunction(model_name=_MODEL)
    client = chromadb.PersistentClient(path=persist_directory)
    collection_name = "system"

    existing = {c.name for c in client.list_collections()}
    needs_index = collection_name not in existing

    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=ef
    )

    if needs_index or collection.count() == 0:
        doc_dir = os.environ["SYSTEM"]
        documents = get_documents(doc_dir)
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            collection.add(
                documents=[d["page_content"] for d in batch],
                metadatas=[d["metadata"] for d in batch],
                ids=[str(i + j) for j in range(len(batch))],
            )

    return VectorStore(collection)
