import atexit
import hashlib
import itertools
import logging
from pprint import pprint

from classifier.result.predict import MODELS
from integrator.serialize import serialize_graph_to_structure
from integrator.states import states
from integrator.tree import Tree
from lib.proximity import set_up_db_from_model
from lib.t import catchtime, indented


@atexit.register
def goodbye():
    print("You are now leaving the Python sector.")


def infer(model_name, t, valid_labels, tree=None, on=None, on_indices=None):
    pre_computed_scores = None

    if isinstance(t, Tree):
        config = MODELS[model_name].config

        if t.all_computed(
            relations=config.relations,
        ):
            print(f"All edges computed for {model_name}")
            return None

        keys, inp = t.pull_batch(
            config.batch_size,
            config.n_pull if config.n_pull else config.n_samples,
            relations=config.relations,
            on=on,
            on_indices=on_indices,
        )
    elif isinstance(t, list):
        if len(t) >= 1:
            if isinstance(t[0], tuple):
                texts = []
                keys = []
                pre_computed_scores = []
                for (n1, n2), score in t:
                    if n1 not in tree.index_text or n2 not in tree.index_text:
                        logging.error(
                            f"node {n1} or {n2} not in tree index: {tree.index_text}"
                        )
                        continue
                    texts.append((tree.index_text[n1], tree.index_text[n2]))
                    keys.append((n1, n2))
                    pre_computed_scores.append(score)
                keys, inp = keys, texts
            else:
                keys, inp = [t], t

    if not inp:
        print(f"All edges already computed for {model_name}")
        return []

    labels, score = MODELS[model_name].predict(inp)
    labels, score = list(
        labels.view(-1, MODELS[model_name].config.n_samples).tolist()
    ), list(score.view(-1, MODELS[model_name].config.n_samples).tolist())

    if pre_computed_scores:
        score = [[s, s] for s in pre_computed_scores]
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


def score(model_name, t, relation, hash_id="default"):
    config = MODELS[model_name].config

    if isinstance(t, Tree):
        if t.all_computed(
            relations=config.relations,
        ):
            print(f"All edges computed for {model_name}")
            return None

        input_dict = list(
            t.pull(
                1,
                relations=config.relations,
            )
        )
        input_dict = {k[0][0]: k[0][1] for k in input_dict}

    elif isinstance(t, list):
        input_dict = {i: i for i in t}

    if not input_dict:
        print(f"All edges already computed for {relation}")
        return []

    vector_store = set_up_db_from_model(
        hash_id + "_" + relation, input_dict, MODELS[model_name], config
    )

    results = []
    for key, value in input_dict.items():
        results.append(
            (
                key,
                vector_store.similarity_search_with_score(
                    value, k=config.get("top_k", 10), search_type="mmr"
                ),
            )
        )

    results = list(
        set(
            [
                (tuple(sorted([key, res.metadata["key"]])), round(score, 4))
                for key, result in results
                for res, score in result
                if score != 0
            ]
        )
    )

    # normalize scores
    scores = [score for _, score in results]
    max_score = max(scores)
    min_score = min(scores)
    results = [
        (key, (score - min_score) / (max_score - min_score)) for key, score in results
    ]

    return results


# classify thesis, antithesis, synthesis
def classifier(t, on=None, on_indices=None):
    return infer(
        "thesis_antithesis_synthesis", t, [1, 2, 3], on=on, on_indices=on_indices
    )


# classify hypernym, hyponym
def hierarchy(t):
    hie = score("hierarchy", t, relation="hie")
    hie_directed = infer("hierarchy", hie, [1, 2], tree=t)

    return hie_directed


# score hypernym, hyponym
def score_hierarchy(t):
    return score("score_hierarchy", t, relation="hie")


def opposite(t):
    return score("opposites", t, relation="ant")


ITERATORS = {}


def update_triangle_graph(t: Tree, i, hash_id):
    if not hash_id in ITERATORS:
        ITERATORS[hash_id] = "hie"
    if not ITERATORS[hash_id] == "end":
        try:
            choice = ITERATORS[hash_id]
            if choice == "hie":
                with catchtime("SUBSUMTION"):
                    if not t.all_computed(relations=["hie"]):
                        hie_score = hierarchy(t)

                        for l, s, k in hie_score:
                            t.add_relation(k[0], k[1], "hie", s[0])
                            t.add_relation(k[1], k[0], "hie", -s[1])

                    else:
                        ITERATORS[hash_id] = "ant"

            elif choice == "syn":
                with catchtime("SYNTHESIS"):
                    lsk = classifier(t, on=t.matrices["ant"], on_indices=t.node_index)
                    if not lsk:
                        ITERATORS[hash_id] = "end"
                    else:
                        for l, s, k in lsk:
                            ant1 = t.get_relation(k[2], k[1], "ant")
                            ant2 = t.get_relation(k[2], k[0], "ant")
                            ant3 = t.get_relation(k[1], k[0], "ant")

                            if not ant1 or not ant2 or not ant3:
                                continue

                            t.add_branching(k[2], k[1], k[0], ant1, ant2 + ant3)

                            t.add_relation(k[2], k[1], "syn_1", ant1)
                            t.add_relation(
                                k[2],
                                k[0],
                                "syn_2",
                                ant2 + ant3,
                            )
                            t.j += 1

            elif choice == "ant":
                with catchtime("ANTITHESIS"):
                    ant_score = opposite(t)
                    if not ant_score:
                        ITERATORS[hash_id] = "syn"
                    else:
                        for (n1, n2), score in ant_score:
                            t.add_relation(n1, n2, "ant", score)
                            t.add_relation(n2, n1, "ant", score)
        except:
            logging.error(f"ERROR IN PREDICTION LOOP {i=} {hash_id=}", exc_info=True)
    t.save_state(i, hash_id)

    return ITERATORS[hash_id]


def make_dialectics(texts, epochs=10, hash_id=None):
    if not isinstance(texts[0], tuple):
        inputs = [(t, t) for t in texts]

    if not hash_id:
        hash_id = "hash_id" + str(inputs).replace("/", "_").replace(".", "_")
        hash_id = hashlib.sha256(hash_id.encode("utf-8")).hexdigest()[:4]
        logging.warning(f"no hash_id provided, using {hash_id} and computed inputs")
    T, i = Tree(inputs), 0

    for _ in range(epochs):
        with catchtime(f"EPOCH {i}"):
            update_triangle_graph(T, i, hash_id)
        print(
            {
                str("_".join(kind)): iterator.get_percentage()
                for kind, iterator in T.iterators.items()
            }
        )

        i += 1

    return T


if __name__ == "__main__":
    hash_id = "eac9fc66ccb6f08a5a5ef26ed20b23dc9d4f84e660daf4d40db84e6f11c17bd0"
    T, i = states[hash_id]

    while True:
        with catchtime(f"update {i}"):
            STATE = update_triangle_graph(T, i, hash_id)

        with catchtime(f"graph  {i}"):
            new_graph = T.max_score_triangle_subgraph(T.graph, return_start_node=True)
        with catchtime(f"serialize {i}"):
            pprint(serialize_graph_to_structure(*new_graph))
        i += 1

        if i > 600:
            break

    path = "texts/cookbook.txt"
    hash_id = "hash_id" + path.replace("/", "_").replace(".", "_")
    not_done = True

    inputs = get_inputs(path)
    T, i = Tree.load_state(hash_id)
    if not i:
        T, i = Tree(list(inputs.items())), 0

    while not_done:
        with catchtime(f"EPOCH {i}"):
            new_graph = update_triangle_graph(T, i, hash_id, return_start_node=True)
        with indented(f"GRAPH " + str(new_graph.__repr__())):
            pass

        # pprint (Tree.serialize_graph_to_structure(*new_graph))
        i += 1
