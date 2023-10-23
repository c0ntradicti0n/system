import logging
from pprint import pprint

import numpy as np
from reader import get_inputs
from tree import Tree

from classifier.predict import MODELS
from lib.t import catchtime


def infer(model_name, tree, valid_labels):
    keys, inp = tree.pull_lz(
        MODELS[model_name].config.batch_size, MODELS[model_name].config.n_samples,
        _score=("h_" if model_name == "hierarchical_2" else "t_")
    )
    labels, score = MODELS[model_name].predict(inp)
    labels, score = list(
        labels.view(-1, MODELS[model_name].config.n_samples).tolist()
    ), list(score.view(-1, MODELS[model_name].config.n_samples).tolist())
    lsk = [
        (l, s, k)
        for l, s, k in zip(labels, score, keys)
        if all(i in list(l) for i in valid_labels)
    ]
    if not lsk:
        return []

    l_s_ks = list(
        list(zip(*sorted(zip(l, s, k), key=lambda x: x[0], reverse=True)))
        for l, s, k in lsk
    )

    return l_s_ks


def classifier(t):
    return infer("tas_3_only", t, [1, 2, 3])


def organizer(t):
    return infer("hierarchical_2", t, [1, 2])


def update_triangle_graph(t: Tree, i, hash, return_start_node=None):
    added = False

    with catchtime("graph"):
        t.draw_graph(
            Tree.max_score_triangle_subgraph(t.graph),
            path=f"{i}.png",
            text_relation=False,
        )

    with catchtime("classifier"):
        try:
            if np.random.choice(["|", "---"], p=[0.7, 0.3]) == "|":
                lsk = organizer(t)
                for l, s, k in lsk:
                    added = "organizer"
                    t.add_relation(k[1], k[0], "sub", h_score=s[0])
            else:
                lsk = classifier(t)
                for l, s, k in lsk:
                    added = "synantithesis"
                    if t.graph.get_edge_data(k[2], k[1], key="ant") and t.graph.get_edge_data(k[2], k[0], key="syn"):
                        t.graph.remove_edge(k[2], k[1], key="ant")

                    t.add_relation(k[2], k[1], "ant", t_score=s[1], trident=t.j)
                    t.add_relation(k[2], k[0], "syn", a_score=s[0], trident=t.j)
                    t.j += 1
        except:
            logging.error(f"error in classifier {i=} {hash=}", exc_info=True)
        if added:
            t.save_state(i, hash)

            print(f"{i} {added}")
            i += 1

    with catchtime("graph"):
        return Tree.max_score_triangle_subgraph(t.graph, return_start_node=return_start_node)


if __name__ == "__main__":
    not_done = True
    # Get the inputs
    inputs = get_inputs("tlp.txt")
    T, i = Tree.load_state("hash")
    if not i:
        T, i = Tree(list(inputs.items())), 0

    while not_done:
        with catchtime("classifier"):
            new_graph = update_triangle_graph(T, i, "hash", return_start_node=True)
        print(new_graph[0].edges)
        pprint (Tree.serialize_graph_to_structure(*new_graph))
        i += 1
