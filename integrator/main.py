import atexit
import logging

import numpy as np

from classifier.predict import MODELS
from integrator.tree import Tree


@atexit.register
def goodbye():
    print("You are now leaving the Python sector.")


from lib.t import catchtime, indented


def infer(model_name, t, valid_labels, on_relation=None):
    if isinstance(t, Tree):
        keys, inp = t.pull_lz(
            min(MODELS[model_name].config.batch_size, len(t.graph.nodes) ** 2),
            MODELS[model_name].config.n_samples,
            _score=("h_" if model_name == "hierarchical_2" else "t_"),
            on_relation=on_relation,
        )
    elif isinstance(t, list):
        keys, inp = [t], t

    labels, score = MODELS[model_name].predict(inp)
    labels, score = list(
        labels.view(-1, MODELS[model_name].config.n_samples).tolist()
    ), list(score.view(-1, MODELS[model_name].config.n_samples).tolist())

    # lsk.. list(labels, score, keys))
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


# classify thesis, antithesis, synthesis
def classifier(t):
    return infer("tas_3_only", t, [1, 2, 3], on_relation="sub")


# classify hypernym, hyponym
def organizer(t):
    return infer("hierarchical_2", t, [1, 2])


def antagonizer(t):
    return infer("ant_wn_only", t, [1, 2], on_relation="ant")


def update_triangle_graph(
    t: Tree, i, hash, return_start_node=None, start_with_sub=False
):
    lsk = []
    i_added = 0
    i_continue = 0
    try:
        if np.random.choice(["|", "---"], p=[0.5, 0.5]) == "|":
            with catchtime("SUBSUMTION"):
                lsk = organizer(t)

            for l, s, k in lsk:
                if any(
                    k[0] == e[1]
                    for e in t.graph.out_edges(k[1], data=True)
                    if e[2].get("relation") == "sub"
                ):
                    i_continue += 1
                    continue

                t.add_relation(k[1], k[0], "sub", h_score=s[1])
                i_added += 1
        else:
            with catchtime("THESIS ANTITHESIS SYNTHESIS"):
                lsk = classifier(t)

            for l, s, k in lsk:
                # make sure that we don't add the same relation twice
                tri_edges = t.computed_i_tri_out_edges(t.graph, k[2], [])
                if any(k[1] == a[1] and k[0] == b[1] for a, b in tri_edges):
                    i_continue += 1
                    continue

                anto = antagonizer([[k[2], k[1]]])

                ant_score = 1 if anto and anto[0] and anto[0][0][0] == 1 else 0

                t.add_relation(k[2], k[1], "ant", a_score=ant_score + s[1], trident=t.j)
                t.add_relation(k[2], k[0], "syn", s_score=s[0], trident=t.j)
                t.j += 1
                i_added += 1
    except:
        logging.error(f"error in classifier {i=} {hash=}", exc_info=True)
    if lsk:
        with indented(f"added {i_added} relations"):
            t.save_state(i, hash)
        with indented(f"discarded {i_continue} relations"):
            pass
    with catchtime("NEW GRAPH"):
        return t.max_score_triangle_subgraph(
            t.graph, return_start_node=return_start_node, start_with_sub=start_with_sub
        )


def make_dialectics(
    texts, epochs=10, hash="test", start_node=None, start_with_sub=False
):
    if not isinstance(texts[0], tuple):
        inputs = [(t, t) for t in texts]

    T, i = Tree(inputs), 0
    if start_node:
        T.params["startNode"] = start_node

    for _ in range(epochs):
        with catchtime(f"EPOCH {i}"):
            new_graph, start_node = update_triangle_graph(
                T, i, hash, return_start_node=True, start_with_sub=start_with_sub
            )
        with indented(
            f"BEST GRAPH {len(new_graph.nodes)} nodes and {len(new_graph.edges)} edges"
        ):
            pass
        i += 1

    return T, (new_graph, start_node)


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
