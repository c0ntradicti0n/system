import os

import Chroma


def set_up_db_from_model(hash, documents, model):
    persist_directory = f"states/{hash}_db/"
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        os.makedirs(persist_directory)

    vector_store = Chroma.from_documents(
        embedding=model,
        documents=documents,
        collection_name=hash,
        persist_directory=persist_directory,
    )
    vector_store.persist()
    return vector_store
