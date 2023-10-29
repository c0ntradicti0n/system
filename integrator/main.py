import atexit
import logging

import numpy as np

from classifier.predict import MODELS
from integrator.tree import Tree


@atexit.register
def goodbye():
    print("You are now leaving the Python sector.")


from lib.t import catchtime, indented


def infer(model_name, tree, valid_labels, on_relation=None):
    keys, inp = tree.pull_lz(
        MODELS[model_name].config.batch_size,
        MODELS[model_name].config.n_samples,
        _score=("h_" if model_name == "hierarchical_2" else "t_"),
        on_relation=on_relation,
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
    return infer("tas_3_only", t, [1, 2, 3], on_relation="sub")


def organizer(t):
    return infer("hierarchical_2", t, [1, 2])


def update_triangle_graph(t: Tree, i, hash, return_start_node=None):
    lsk = []
    try:
        if np.random.choice(["|", "---"], p=[0.5, 0.5]) == "|":
            with catchtime("SUBSUMTION"):
                lsk = organizer(t)

            for l, s, k in lsk:
                t.add_relation(k[1], k[0], "sub", h_score=s[0])
        else:
            with catchtime("THESIS ANTITHESIS SYNTHESIS"):
                lsk = classifier(t)

            for l, s, k in lsk:
                if t.graph.get_edge_data(
                    k[2], k[1], key="ant"
                ) and t.graph.get_edge_data(k[2], k[0], key="syn"):
                    t.graph.remove_edge(k[2], k[1], key="ant")

                t.add_relation(k[2], k[1], "ant", t_score=s[1], trident=t.j)
                t.add_relation(k[2], k[0], "syn", a_score=s[0], trident=t.j)
                t.j += 1
    except:
        logging.error(f"error in classifier {i=} {hash=}", exc_info=True)
    if lsk:
        with indented(f"added {len(lsk)} relations"):
            t.save_state(i, hash)

    with catchtime("NEW GRAPH"):
        return t.max_score_triangle_subgraph(
            t.graph, return_start_node=return_start_node
        )


if __name__ == "__main__":
    hash = "b3607805fa4d6ef807e825b44c081446dee7a5b14a796981262f9234686d4ff9"
    T, i = Tree.load_state(hash)

    while True:
        with catchtime(f"EPOCH {i}"):
            new_graph = update_triangle_graph(T, i, hash, return_start_node=True)
        with indented(f"GRAPH " + str(new_graph.__repr__())):
            pass

        # pprint (Tree.serialize_graph_to_structure(*new_graph))
        i += 1

    path = "texts/cookbook.txt"
    hash = "hash" + path.replace("/", "_").replace(".", "_")
    not_done = True

    inputs = get_inputs(path)
    T, i = Tree.load_state(hash)
    if not i:
        T, i = Tree(list(inputs.items())), 0

    while not_done:
        with catchtime(f"EPOCH {i}"):
            new_graph = update_triangle_graph(T, i, hash, return_start_node=True)
        with indented(f"GRAPH " + str(new_graph.__repr__())):
            pass

        # pprint (Tree.serialize_graph_to_structure(*new_graph))
        i += 1
