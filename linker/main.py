import logging
import os
import re

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from sklearn.metrics.pairwise import cosine_similarity

from lib.doc import get_documents
from lib.embedding import get_embeddings
from lib.t import catchtime


def write_generator(gen, tmp_dir="", strip_path=""):
    for path, content in gen:
        if strip_path:
            path = path.replace(strip_path, "")
        p = os.path.join(tmp_dir, path)
        d = os.path.dirname(p)
        if tmp_dir:
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(p) and not tmp_dir:
            raise (ValueError(f"File {p} does not exist and so can't be changed."))
        with open(p, "w", encoding="utf-8") as file:
            file.write(content)


def remove_links(text, pattern):
    # Use the named group 'text' to keep the text of the link while removing the URL
    return pattern.sub(r"\g<text>", text)


def link_texts(
    document_dir,
    vector_store,
    n=3,
    relevance_threshold=0.55,
    batch_size=32,
    punctuation_penalty=0.5,
):
    all_files = []
    for root, _, files in os.walk(document_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    all_files = [x for x in all_files if ".git" not in x and x.endswith(".md")]

    link_pattern = re.compile(r"\[(?P<text>[^\]]+)\]\(([^)]+)\)")

    for file_path in all_files:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
        if not text or len(text) < 150:
            logging.info(f"Skipping empty file {file_path}")
            continue

        text = remove_links(text, link_pattern)

        doc_embedding = get_embeddings([text])

        segments = link_pattern.split(text)
        processed_segments = []

        for segment in segments:
            lines = segment.split("\n")
            processed_lines = []

            for line in lines:
                words = line.split()
                i = 0
                linked_text = []
                n_grams = []
                n_gram_indices = []

                while i < len(words):
                    if i + n > len(words):
                        n_gram = " ".join(words[i:])
                    else:
                        n_gram = " ".join(words[i : i + n])
                    n_grams.append(n_gram)
                    n_gram_indices.append(i)
                    i += n

                    # If batch is full or all n-grams are added, process the batch
                    if len(n_grams) == batch_size or i >= len(words):
                        n_gram_embeddings = get_embeddings(n_grams)

                        similarities = cosine_similarity(
                            doc_embedding, n_gram_embeddings
                        ).flatten()  # Flatten the similarities array

                        for j, similarity in enumerate(similarities):
                            # Apply penalty if n-gram contains sentence-ending punctuation
                            if any(
                                punct in n_grams[j]
                                for punct in [",", ";", ":", ".", "!", "?"]
                            ):
                                similarity *= punctuation_penalty

                            if similarity >= relevance_threshold:
                                results = vector_store.search(
                                    n_grams[j], k=1, search_type="mmr"
                                )
                                if results:
                                    best_match = results[0]
                                    link_path = best_match.metadata["file_path"]
                                    link_path = link_path.replace(".md", "").replace(
                                        " ", "%20"
                                    )
                                    linked_text.append(f"[{n_grams[j]}]({link_path})")
                                else:
                                    linked_text.extend(
                                        words[n_gram_indices[j] : n_gram_indices[j + 1]]
                                        if j + 1 < len(n_gram_indices)
                                        else words[n_gram_indices[j] :]
                                    )
                            else:
                                linked_text.extend(
                                    words[n_gram_indices[j] : n_gram_indices[j + 1]]
                                    if j + 1 < len(n_gram_indices)
                                    else words[n_gram_indices[j] :]
                                )

                        # Clear the batch
                        n_grams = []
                        n_gram_indices = []
                processed_lines.append(" ".join(linked_text))

            processed_segments.append("\n".join(processed_lines))

        final_text = "".join(processed_segments)

        with open(file_path, "w", encoding="utf-8") as file:
            yield file_path, final_text


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

if __name__ == "__main__":
    with catchtime("linking"):
        for i in range(3):
            write_generator(
                link_texts(os.environ["SYSTEM"], vector_store)
                # strip_path=os.environ["SYSTEM"]
            )
