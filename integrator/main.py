import atexit
import hashlib
import itertools
import logging

from classifier.result.predict import MODELS
from integrator.states import states
from integrator.tree import Tree
from lib.max_islice import maxislice
from lib.t import catchtime, indented


@atexit.register
def goodbye():
    print("You are now leaving the Python sector.")


def infer(model_name, t, valid_labels):
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
        )
    elif isinstance(t, list):
        keys, inp = [t], t

    if not inp:
        print(f"All edges already computed for {model_name}")
        return []

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
        )
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


def update_triangle_graph(t: Tree, i, hash_id):
    if not hash_id in ITERATORS:
        ITERATORS[hash_id] = "hie"
    if not ITERATORS[hash_id] == "end":
        try:
            choice = ITERATORS[hash_id]
            if choice == "hie":
                with catchtime("SUBSUMTION"):
                    lsk = hierarchy(t)
                    if not lsk:
                        ITERATORS[hash_id] = "ant"
                    else:
                        for l, s, k in lsk:
                            score = score_hierarchy(
                                [t.get_text(k[1]), t.get_text(k[0])]
                            )[0][1]

                            t.add_relation(k[1], k[0], "hie", -score)
                            t.add_relation(k[0], k[1], "hie", score)

            elif choice == "syn":
                with catchtime("SYNTHESIS"):
                    lsk = classifier(t)
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
        except:
            logging.error(f"error in classifier {i=} {hash_id=}", exc_info=True)
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
        print (   {str("_".join(kind)): iterator.get_percentage() for kind, iterator in T.iterators.items()})

        i += 1

    return T


if __name__ == "__main__":
    hash_id = "f22421485e3987d60b5f98a8615413f4638587f56c783380c6810baf6fb4c457"
    T, i = states[hash_id]

    while True:
        with catchtime(f"EPOCH {i}"):
            STATE = update_triangle_graph(T, i, hash_id)

        # pprint (Tree.serialize_graph_to_structure(*new_graph))
        i += 1

        if i > 40:
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
