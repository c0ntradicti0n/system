from langchain.embeddings import SentenceTransformerEmbeddings


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def get_embedding(text):
    return embeddings.embed_documents(text)


