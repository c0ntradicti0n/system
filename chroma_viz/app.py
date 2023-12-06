from langchain.vectorstores import Chroma
vectordb = Chroma.from_documents(data, embeddings, ids)

from chromaviz import visualize_collection
visualize_collection(vectordb._collection)