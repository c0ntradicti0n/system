import os
import time

import chromadb
import requests

from lib.doc import get_documents


class OpenAICompatibleEmbedding:
    def __init__(self, api_key, model, base_url):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def name():
        return "openai_compatible"

    def embed_query(self, input):
        return self.__call__(input)

    def embed_documents(self, input):
        return self.__call__(input)

    def __call__(self, input):
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": input},
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            item["embedding"]
            for item in sorted(data["data"], key=lambda x: x["index"])
        ]


class LocalEmbedding:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        self.ef = SentenceTransformerEmbeddingFunction(model_name=model_name)

    @staticmethod
    def name():
        return "local_sentence_transformer"

    def embed_query(self, input):
        return self.ef(input)

    def embed_documents(self, input):
        return self.ef(input)

    def __call__(self, input):
        return self.ef(input)


def get_embedding_function():
    provider = os.environ.get("EMBEDDING_PROVIDER", "local")
    if provider == "api":
        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        return OpenAICompatibleEmbedding(api_key, model, base_url)
    return LocalEmbedding()


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
    ef = get_embedding_function()
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
        total = len(documents)
        print(f"indexing {total} document chunks from {doc_dir}", flush=True)
        batch_size = 100
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            print(f"  batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} docs)", flush=True)
            t0 = time.time()
            for d in batch[:3]:
                print(f"    metadata keys: {list(d['metadata'].keys())}", flush=True)
            collection.add(
                documents=[d["page_content"] for d in batch],
                metadatas=[d["metadata"] for d in batch],
                ids=[str(i + j) for j in range(len(batch))],
            )
            print(f"  batch done in {time.time()-t0:.1f}s", flush=True)
        print(f"indexing complete — {collection.count()} total vectors", flush=True)

    return VectorStore(collection)
