import gc
import os
from pprint import pprint

from lib.t import catchtime
from lib.vector_store import get_vector_store

with catchtime("init"):
    persist_directory = "/chroma"
    os.system("pwd")
    vector_store = get_vector_store(persist_directory)


def search(query, top_k=5, filter_path=""):
    gc.collect()

    if filter_path:
        filter = {filter_path: {"$eq": filter_path}}
    else:
        filter = None

    results = vector_store.search(query, k=top_k, search_type="mmr", filter=filter)
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

    with catchtime("search"):
        pprint(search("best food?", filter_path="2"))
