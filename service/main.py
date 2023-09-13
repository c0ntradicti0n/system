import os
from pprint import pprint
from time import perf_counter

import regex
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


class catchtime:
    def __init__(self, task):
        self.task = task

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f"Time {self.task}: {self.time:.3f} seconds"
        print(self.readout)


with catchtime("init"):
    def get_filename_without_extension(path):
        return os.path.splitext(os.path.basename(path))[0]


    def get_documents(document_dir):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        all_files = []
        for root, _, files in os.walk(document_dir):
            for file in files:
                all_files.append(os.path.join(root, file))

        all_files = [x for x in all_files if ".git" not in x]

        documents = []
        for file_path in all_files:
            if not file_path.endswith(".md"):
                continue
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            topic = get_filename_without_extension(file_path)
            content = topic + "\n\n" + text
            try:
                path = regex.match(r"^[\/1-3_]*", file_path.replace(document_dir, "")).group(0)
            except:
                raise ValueError(f"Error parsing path for {file_path}")
            clean_path = path.replace("/", "")
            docs = text_splitter.create_documents(
                [content], metadatas=[
                    {
                     "file_path": file_path.replace(document_dir, ""),
                     "path": path,
                     ** {
                         clean_path[:i]: clean_path[:i]
                         for i in range(clean_path.__len__()) }
                         }
                ])
            documents.extend(docs)
        return documents


    persist_directory = ".chroma"
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


def search(query, top_k=10, filter_path=""):
    if filter_path:
        filter =  {filter_path:{'$eq': filter_path}}
    else:
        filter = None

    results = vector_store.search(query, top_k=top_k, search_type="mmr", filter=filter)
    return results


if __name__ == "__main__":
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
