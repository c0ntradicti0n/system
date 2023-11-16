import os

from integrator.main import make_dialectics
from lib.helper import OutputLevel, tree
from lib.nested import extract_values


def recompute_tree(path, start_node, epochs=10, start_with_sub=False):
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
    os.environ["MODELS_CONFIG"] = "../classifier/"
    os.environ["REDIS_HOST"] = "localhost"
    os.system("rm -rf ./states/test")

    t, (g, best_start_node) = make_dialectics(
        texts, epochs=epochs, start_node=start_node, start_with_sub=start_with_sub
    )
    return g, t


def from_texts(texts, start_node=None, epochs=10, start_with_sub=False):
    t, (g, best_start_node) = make_dialectics(
        texts, epochs=epochs, start_node=start_node, start_with_sub=start_with_sub
    )
    return g, t
