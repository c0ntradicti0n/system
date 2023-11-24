import os

from integrator.main import make_dialectics
from lib.helper import OutputLevel, tree
from lib.nested import extract_values


def recompute_tree(path, start_node=None, epochs=10, start_with_sub=False):
    file_dict = tree(
        basepath="../../dialectics/",
        startpath=path,
        format="json",
        keys=path.split("/"),
        info_radius=1,
        exclude=[".git", ".git.md", ".gitignore", ".DS_Store", ".idea"],
        pre_set_output_level=OutputLevel.FILENAMES,
        prefix_items=True,
        depth=os.environ.get("DEPTH", 4),
    )

    print(file_dict)
    texts = extract_values(file_dict)
    texts = [text.replace(".md", "") for text in texts]

    os.system("echo $MODELS_CONFIG")
    os.system("echo $REDIS_HOST")
    os.system("rm -rf ./states/test")
    os.environ["REDIS_HOST"] = "localhost"

    t = make_dialectics(texts, epochs=epochs)
    g, _ = t.max_score_triangle_subgraph(
        t.graph,
        start_node=start_node,
        return_start_node=True,
        start_with_sub=start_with_sub,
    )

    return g, t


def from_texts(texts, epochs=10, start_node=None, start_with_sub=False):
    t = make_dialectics(texts, epochs=epochs)
    g, _ = t.max_score_triangle_subgraph(
        t.graph, return_start_node=True, start_with_sub=start_with_sub
    )
    return g, t


if __name__ == "__main__":
    g, t = from_texts(
        [
            "plus and minus",
            "multiplication and division",
            "exponential and logarithm",
            "addition",
            "subtraction",
            "multiplication",
            "division",
            "exponent",
            "root",
            "neutral element of addition and subtraction is 0",
            "neutral element of multiplication and division is 1",
            "basis of exponent and root",
        ],
        epochs=17,
        start_node="linear operations",
        start_with_sub=True,
    )
    t.dump_graph("calc", t.graph, "calc")
    g, start_node = t.max_score_triangle_subgraph(
        t.graph,
        return_start_node=True,
        start_with_sub=True,
        start_node="linear operations",
    )
    t.draw_graph(g, root=start_node, path="calc.png")
