import os

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from lib.doc import get_documents


def get_vector_store(persist_directory):
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

    return vector_store
