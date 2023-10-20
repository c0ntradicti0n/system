import os

from langchain.embeddings import SentenceTransformerEmbeddings

from lib.shape import view_shape

d = {}

import redis
import pickle


class RedisEmbedder:
    def __init__(self, model_name, host=os.environ.get("REDIS_HOST", 'redis'), port=6379, db=0):
        self.embedder = SentenceTransformerEmbeddings(model_name=model_name)

        # Connect to Redis
        self.redis_client = redis.StrictRedis(host=host, port=port, db=db)

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            # Use the text itself as the key (or a hash of the text if it's too long)
            key = text

            # Check if the embedding for the text exists in Redis
            serialized_embedding = self.redis_client.get(key)

            if serialized_embedding:
                embedding = pickle.loads(serialized_embedding)
            else:
                # Compute the embedding
                embedding = self.embedder.embed_documents([text])[0]
                # Serialize and store the embedding in Redis
                self.redis_client.set(key, pickle.dumps(embedding))

            embeddings.append(embedding)

        return embeddings


def embedder(model_name="BAAI/bge-large-en-v1.5"):
    if model_name in d:
        return d[model_name]
    e = RedisEmbedder(model_name=model_name)
    d[model_name] = e
    return e


def get_embeddings(texts, config):
    r = embedder(config.embedding_model).embed_documents(view_shape(texts, (-1,)))
    if not len(r[0]) == config.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: {len(r[0])} != {config.embedding_dim}"
        )
    return r
