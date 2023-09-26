from langchain.embeddings import SentenceTransformerEmbeddings

from integrator import config

embeddings = SentenceTransformerEmbeddings(model_name="barisaydin/bge-large-en")


def get_embeddings(texts):
    r = embeddings.embed_documents(texts)
    if not len(r[0]) == config.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: {len(r[0])} != {config.embedding_dim}"
        )
    return r
