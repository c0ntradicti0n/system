import atexit
import itertools
import logging

from classifier.result.predict import MODELS
from integrator.tree import Tree
from lib.t import catchtime, indented


@atexit.register
def goodbye():
    print("You are now leaving the Python sector.")


def infer(model_name, t, valid_labels, on_relation=None, for_relation=None):
    if isinstance(t, Tree):
        config = MODELS[model_name].config
        keys, inp = t.pull_lz(
            min(config.batch_size, len(t.graph.nodes) ** 2),
            config.n_pull if config.n_pull else config.n_samples,
            _score=("h_" if model_name == "hierarchical_2" else "t_"),
            on_relation=on_relation,
            for_relation=for_relation,
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


def score(model_name, t, relation):
    if isinstance(t, Tree):
        not_yet_computed = t.missing_edges(relation)
        keys = [(n1, n2) for n1, n2 in not_yet_computed]
        inp = [
            (t.graph.nodes[n1]["text"], t.graph.nodes[n2]["text"])
            for n1, n2 in not_yet_computed
        ]
    elif isinstance(t, list):
        keys, inp = [t], [t]

    if not inp:
        print(f"All edges already computed for {relation}")
        return []

    result = MODELS[model_name].predict(inp)

    result = list(zip(keys, result))

    return result


# classify thesis, antithesis, synthesis
def classifier(t):
    return infer("thesis_antithesis_synthesis", t, [1, 2, 3])


# classify hypernym, hyponym
def hierarchy(t):
    return infer("hierarchy", t, [1, 2])


# score hypernym, hyponym
def score_hierarchy(t):
    return score("score_hierarchy", t, relation="hie")


def opposite(t):
    return score("opposites", t, relation="ant")


ITERATORS = {}


def update_triangle_graph(
    t: Tree, i, hash, return_start_node=None, start_with_sub=False
):
    if not hash in ITERATORS:
        ITERATORS[hash] = itertools.cycle(["hie", "ant", "syn"])

    i_added, i_continue = 0, 0
    try:
        choice = next(ITERATORS[hash])
        if choice == "hie":
            with catchtime("SUBSUMTION"):
                lsk = hierarchy(t)

                for l, s, k in lsk:
                    score = score_hierarchy([t.get_text(k[1]), t.get_text(k[0])])[0][1]

                    t.add_relation(k[1], k[0], "hie", h_score=-score)

                    # negative score for other direction
                    t.add_unique_relation(k[0], k[1], "hie", h_score=score)

                    i_added += 1
        elif choice == "syn":
            with catchtime("SYNTHESIS"):
                lsk = classifier(t)

            for l, s, k in lsk:
                ant1 = t.get_relation(k[2], k[1], "ant")
                ant2 = t.get_relation(k[2], k[0], "ant")
                ant3 = t.get_relation(k[1], k[0], "ant")

                if not ant1 or not ant2 or not ant3:
                    i_continue += 1
                    continue

                t.add_relation(
                    k[2], k[1], "syn_1", A_score=ant1["a_score"], trident=t.j
                )
                t.add_relation(
                    k[2],
                    k[0],
                    "syn_2",
                    T_score=(ant2["a_score"] + ant3["a_score"]) / 2,
                    trident=t.j,
                )
                t.j += 1
                i_added += 1

        elif choice == "ant":
            with catchtime("ANTITHESIS"):
                ant_score = opposite(t)

                for (n1, n2), s in ant_score:
                    n_added = t.add_unique_relation(n1, n2, "ant", a_score=s)
                    n_added += t.add_unique_relation(n2, n1, "ant", a_score=s)
                    i_added += n_added

    except:
        logging.error(f"error in classifier {i=} {hash=}", exc_info=True)

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
