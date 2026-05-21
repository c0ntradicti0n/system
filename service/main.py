import gc
import os
import threading
from pprint import pprint

from lib.t import catchtime
from lib.vector_store import VectorStore, get_embedding_function

# ── Fast init: open ChromaDB immediately, index docs in background ──
print("init: opening chroma...", flush=True)
import chromadb

ef = get_embedding_function()
client = chromadb.PersistentClient(path="/chroma")
collection_name = "system"
existing = {c.name for c in client.list_collections()}
needs_index = collection_name not in existing
collection = client.get_or_create_collection(
    name=collection_name, embedding_function=ef
)

vector_store = None

if needs_index or collection.count() == 0:
    doc_dir = os.environ["SYSTEM"]
    from lib.doc import get_documents as _get_documents

    def _index_worker():
        global vector_store
        print(f"indexing {doc_dir} in background...", flush=True)
        documents = _get_documents(doc_dir)
        total = len(documents)
        print(f"indexing {total} document chunks from {doc_dir}", flush=True)
        batch_size = 100
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            collection.add(
                documents=[d["page_content"] for d in batch],
                metadatas=[d["metadata"] for d in batch],
                ids=[str(i + j) for j in range(len(batch))],
            )
        cnt = collection.count()
        print(f"indexing complete — {cnt} total vectors", flush=True)
        vector_store = VectorStore(collection)

    threading.Thread(target=_index_worker, daemon=True).start()
else:
    vector_store = VectorStore(collection)
    print(f"collection has {collection.count()} vectors, skipping index", flush=True)


def search(query, top_k=5, filter_path=""):
    global vector_store
    gc.collect()

    if vector_store is None:
        return []

    filter = None
    if filter_path:
        filter = {filter_path: {"$eq": filter_path}}

    results = vector_store.search(query, k=top_k, search_type="mmr", filter=filter)
    return results


if __name__ == "__main__":
    import time
    time.sleep(5)
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
