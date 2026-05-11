import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List

import chromadb
import redis
import torch
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class CustomEmbeddingWrapper:
    def __init__(self, sentence_model_name, custom_model, host=None, port=6379, db=0):
        # Initialize the Sentence Transformer model
        self.sentence_embedder = SentenceTransformer(sentence_model_name)

        # Custom model that takes sentence embeddings as input
        self.custom_model = custom_model

        # Connect to Redis
        self.redis_client = redis.StrictRedis(
            host=host if host else os.environ.get("REDIS_HOST", "redis"),
            port=port,
            db=db,
        )

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            key = text  # Or use a hash of the text

            # Check Redis cache
            serialized_embedding = self.redis_client.get(key)

            if serialized_embedding:
                embedding = pickle.loads(serialized_embedding)
            else:
                # Generate embedding using Sentence Transformer
                embedding = self.sentence_embedder.encode([text])[0]
                # Cache the embedding
                self.redis_client.set(key, pickle.dumps(embedding))

            # Process embedding with the custom model
            custom_embedding = self.custom_model.embed(
                torch.tensor(embedding).unsqueeze(0)
            )

            embeddings.append(custom_embedding.tolist()[0][0])

        return embeddings

    def embed_query(self, query):
        # Check Redis cache first
        key = query  # Or use a hash of the query
        serialized_embedding = self.redis_client.get(key)

        if serialized_embedding:
            embedding = pickle.loads(serialized_embedding)
        else:
            # Generate embedding using Sentence Transformer
            embedding = self.sentence_embedder.encode([query])[0]
            # Optionally cache the embedding
            self.redis_client.set(key, pickle.dumps(embedding))

        # Process embedding with the custom model
        custom_embedding = self.custom_model.embed(torch.tensor(embedding).unsqueeze(0))

        return custom_embedding.tolist()[0][0]


def _split_text(text: str, chunk_size: int = 3000) -> List[str]:
    """Split text into chunks of at most chunk_size characters."""
    return [text[i : i + chunk_size] for i in range(0, max(len(text), 1), chunk_size)]


class ChromaVectorStore:
    """Thin wrapper around a chromadb collection that mimics the langchain Chroma API."""

    def __init__(self, collection, embedding_fn):
        self._collection = collection
        self._embedding_fn = embedding_fn

    def similarity_search_with_score(self, query: str, k: int = 10, search_type=None):
        query_embedding = self._embedding_fn([query])
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        docs_with_scores = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Convert distance to a similarity-like score (1 - cosine_distance)
            score = round(1.0 - dist, 6)
            docs_with_scores.append((Document(page_content=doc, metadata=meta), score))
        return docs_with_scores


def set_up_db_from_model(hash, input_dict, model, config):
    sentence_embedding = CustomEmbeddingWrapper(
        config.embedding_model, model[model.active_model_name]
    )

    # Build chunks from input_dict values
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    idx = 0
    for key, text in input_dict.items():
        for chunk in _split_text(str(text), chunk_size=3000):
            documents.append(chunk)
            metadatas.append({"key": key})
            ids.append(f"{hash}_{idx}")
            idx += 1

    # Use an in-memory chromadb client (no persist needed for transient state)
    client = chromadb.EphemeralClient()
    # Sanitise collection name: chromadb requires [a-zA-Z0-9_-] and length 3-63
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in hash)[:63]
    if len(safe_name) < 3:
        safe_name = safe_name.ljust(3, "_")

    try:
        client.delete_collection(safe_name)
    except Exception:
        pass

    collection = client.create_collection(safe_name, metadata={"hnsw:space": "cosine"})

    if documents:
        embeddings = sentence_embedding.embed_documents(documents)
        # embed_documents returns raw lists/floats; convert to nested list if needed
        if embeddings and not isinstance(embeddings[0], list):
            embeddings = [[e] if not hasattr(e, "__iter__") else list(e) for e in embeddings]
        collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)

    def _embed_query(texts):
        embs = sentence_embedding.embed_documents(texts)
        if embs and not isinstance(embs[0], list):
            embs = [[e] if not hasattr(e, "__iter__") else list(e) for e in embs]
        return embs

    return ChromaVectorStore(collection, _embed_query)
