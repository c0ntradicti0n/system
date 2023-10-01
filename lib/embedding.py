from langchain.embeddings import SentenceTransformerEmbeddings

d = {}
def embedder(model_name="BAAI/bge-large-en-v1.5"):
    if model_name in d:
        return d[model_name]
    e = SentenceTransformerEmbeddings(model_name=model_name)
    d[model_name] = e
    return e


def get_embeddings(texts, config):
    r = embedder(config.embedding_model).embed_documents(texts)
    if not len(r[0]) == config.embedding_dim:
        raise ValueError(
            f"Embedding dimension mismatch: {len(r[0])} != {config.embedding_dim}"
        )
    return r
