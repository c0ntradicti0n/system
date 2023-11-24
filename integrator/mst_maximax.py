import copy
import math
import pickle
from pprint import pprint

import networkx as nx

from integrator.mst import edge_priority
from integrator.serialize import serialize_graph_to_structure
from lib.dict_diff import dict_diff
from lib.draw_graph import draw_graph
from lib.t import catchtime, indented


def find_sub_edge_score(G, u, v):
    for n1, n2, attr in G.out_edges(u, data=True):
        if n2 == v and attr["relation"] == "hie":
            return attr["hie_score"]
    return 0.3


def computed_sub_out_edges(G, v, visited, parents):
    return list(
        sorted(
            [
                (
                    n1,
                    n2,
                    {
                        **attr,
                        "hie_score": math.prod(
                            [
                                attr["hie_score"],
                                *(
                                    find_sub_edge_score(G, parent, n2)
                                    for parent in parents
                                ),
                            ]
                        ),
                    },
                )
                for n1, n2, attr in G.out_edges(v, data=True)
                if attr["relation"] == "hie" and n2 not in visited
            ],
            key=lambda x: x[2]["hie_score"],
            reverse=True,
        )
    )


def computed_i_tri_out_edges(G, v, visited):
    if v not in G.graph["branches"]:
        return []
    edges = [
        ((n1, score1), (n2, score1))
        for (n1, n2), (score1, score2) in G.graph["branches"][v].items()
    ]
    edges = [
        pair
        for pair in edges
        if pair[0][0] not in visited and pair[0][1] not in visited
    ]
    try:
        result = sorted(
            edges,
            key=lambda x: x[0][1] + x[1][1],
            reverse=True,
        )
    except Exception:
        raise
    return result


def determine_edge_type(edge):
    # Assuming edge attributes include 'relation'
    relation = edge[2].get("relation", "")
    if relation == "ant":
        return "ant"
    elif relation == "syn":
        return "syn"
    elif relation == "hie":
        return "hie"
    return "unknown"


def all_neighbors_visited(node, visited, G):
    for neighbor in G.neighbors(node):
        if neighbor not in visited:
            return False
    return True


def score_of_mst(mst):
    score = 0
    for edge in mst.edges(data=True):
        score += edge_priority(edge, determine_edge_type(edge))
    return score


def merge_graphs(graphs):
    merged_graph = nx.DiGraph()
    for graph in graphs:
        for node, data in graph.nodes(data=True):
            if node not in merged_graph:
                merged_graph.add_node(node, **data)
        for u, v, data in graph.edges(data=True):
            if not merged_graph.has_edge(u, v):
                merged_graph.add_edge(u, v, **data)
    return merged_graph


def maximax(node, depth, is_maximizing_player, G, visited, current_mst, parents=None):
    if not parents:
        parents = []

    def finale():
        return score_of_mst(current_mst), current_mst

    if depth == 0 or all_neighbors_visited(node, visited, G):
        return finale()
    best_mst = None
    max_eval = float("-inf")

    original_mst = copy.deepcopy(current_mst)
    original_visited = copy.deepcopy(visited)

    if is_maximizing_player:
        sub_edges = computed_sub_out_edges(G, node, visited, parents=parents)
        for node, child, attr in sub_edges:
            current_mst = copy.deepcopy(original_mst)
            visited = copy.deepcopy(original_visited)

            if child not in visited:
                visited.add(child)
                current_mst.add_edge(node, child, **attr)
                eval, mst_candidate = maximax(
                    child, depth - 1, False, G, visited, copy.deepcopy(current_mst)
                )
                if eval > max_eval:
                    max_eval = eval
                    best_mst = mst_candidate
                visited.remove(child)
                current_mst.remove_edge(node, child)
        if max_eval == float("-inf"):
            return finale()
        return max_eval, best_mst

    else:
        tri_edges = computed_i_tri_out_edges(G, node, visited)
        max_eval = float("-inf")
        best_mst = None

        for edge_a, edge_b in tri_edges:
            child_a, score1 = edge_a
            child_b, score2 = edge_b
            if child_a == child_b:
                print("same child")
                continue

            visited = copy.deepcopy(original_visited)
            current_mst = copy.deepcopy(original_mst)

            if child_a not in visited and child_b not in visited:
                visited.add(child_a)
                visited.add(child_b)
                visited.add(node)

                current_mst.add_edge(node, child_a, relation="syn_1", A_score=score1)
                current_mst.add_edge(node, child_b, relation="syn_2", T_score=score2)

                eval_b, mst_b = maximax(
                    child_b,
                    depth - 1,
                    True,
                    G,
                    visited,
                    copy.deepcopy(current_mst),
                    parents=[node, *parents],
                )

                eval_a, mst_a = maximax(
                    child_a,
                    depth - 1,
                    True,
                    G,
                    set(mst_b.nodes),
                    copy.deepcopy(mst_b),
                    parents=[node, *parents],
                )

                eval_0, mst_0 = maximax(
                    node,
                    depth - 1,
                    True,
                    G,
                    set(mst_a.nodes),
                    copy.deepcopy(mst_a),
                    parents=[node, *parents],
                )
                current_mst.remove_edge(node, child_a)
                current_mst.remove_edge(node, child_b)
                visited.remove(node)
                visited.remove(child_a)
                visited.remove(child_b)

                merged_mst = merge_graphs([mst_0, mst_a, mst_b])

                combined_score = score_of_mst(merged_mst)
                if combined_score > max_eval:
                    max_eval = combined_score
                    best_mst = merged_mst

        if max_eval == float("-inf"):
            return finale()
        return max_eval, best_mst


def construct_mst(G, root, max_depth, start_with_sub=False):
    with catchtime("computing maximax"):
        initial_mst = nx.DiGraph()
        visited = {root}

        best_score, best_mst = maximax(
            root, max_depth, start_with_sub, G, visited, initial_mst
        )
        with indented(f"Best score: {best_score}"):
            pass

        for node in best_mst.nodes:
            best_mst.nodes[node].update(G.nodes[node])

        return best_mst


def scenario_calc(G):
    with open("calc", "rb") as f:
        g = pickle.load(f)
    G.update(g)
    # print sub score as a matrix

    return (
        {i: n["text"] for i, n in G.nodes(data=True)},
        "plus and minus",
        {
            1: {1: {".": "basis of exponent and root"}, ".": "plus and minus"},
            2: {1: {".": "multiplication and division"}, ".": "addition"},
            3: {
                1: {".": "neutral element of addition and subtraction is 0"},
                2: {".": "neutral element of multiplication and division is 1"},
                3: {".": "division"},
                ".": "exponential and logarithm",
            },
        },
    )


def scenario_1(G):
    N = {
        1: "Be",
        2: "Not to be",
        3: "Becoming",
        4: "Same and not the Same of Being and Non-Being",
        5: "Coming-into-Being and Ceasing-to-Be",
        6: "Sublation",
    }
    G.add_edge(N[1], N[2], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[4], N[5], A_score=1.1, relation="syn_1")  # Antonym edge
    G.add_edge(N[1], N[3], T_score=2, relation="syn_2")  # Synthesis edge
    G.add_edge(N[4], N[6], T_score=2.1, relation="syn_2")  # Synthesis edge
    G.add_edge(N[3], N[4], hie_score=3, relation="hie")  # Hypernym/Hyponym edge

    G.graph["branches"] = {
        N[1]: {
            (N[2], N[3]): (1.5, 1.4),
        },
        N[4]: {
            (N[5], N[6]): (1.6, 1.7),
        },
    }

    expectation = {
        1: {".": "Be"},
        2: {".": "Not to be"},
        3: {
            1: {".": "Same and not the " "Same of Being and Non-Being"},
            2: {".": "Coming-into-Being and Ceasing-to-Be"},
            3: {".": "Sublation"},
            ".": "Becoming",
        },
    }

    return N, N[1], expectation


def scenario_2(G):
    N = {
        0: "mathematics",
        21: "arithmetic",
        1: "plus",
        2: "minus",
        3: "plus minus n is 0",
        4: "times",
        5: "divided by",
        6: "times divided n is 1",
        7: "exponent",
        8: "root",
        9: "root of exponent is n",
        10: "line calculation",
        11: "point calculation",
        12: "repeated calculation",
        20: "number theory",
        14: "natural number",
        15: "integer",
        15.5: "float",
        16: "rational number",
        17: "real number",
        18: "complex number",
        19: "number",
        24: "prime numbers",
        25: "even numbers",
        26: "odd numbers",
        27: "pi",
        28: "e",
        22: "algebra",
    }

    """29: "golden ratio",
    30: "infinity",
    31: "zero",
    32: "one",

    33: "variable",
    34: "constant",
    35: "function",
    36: "equation",
    37: "inequality",
    38: "polynomial",
    39: "expression","""

    G.add_edge(N[0], N[21], hie_score=30, relation="hie")  # Hypernym/Hyponym edge
    G.add_edge(N[21], N[20], A_score=10, relation="syn_1")  # Antonym edge
    G.add_edge(N[21], N[22], T_score=20, relation="syn_2")  # Synthesis edge

    G.add_edge(N[21], N[10], hie_score=30, relation="hie")  # Hypernym/Hyponym edge

    G.add_edge(N[1], N[2], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[1], N[3], T_score=2, relation="syn_2")  # Synthesis edge
    G.add_edge(N[4], N[5], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[4], N[6], T_score=2, relation="syn_2")  # Synthesis edge
    G.add_edge(N[7], N[8], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[7], N[9], T_score=2, relation="syn_2")  # Synthesis edge
    G.add_edge(N[10], N[11], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[10], N[12], T_score=2, relation="syn_2")  # Synthesis edge
    G.add_edge(N[10], N[1], hie_score=3, relation="hie")  # Hypernym/Hyponym edge
    G.add_edge(N[11], N[4], hie_score=3, relation="hie")  # Hypernym/Hyponym edge
    G.add_edge(N[12], N[7], hie_score=3, relation="hie")  # Hypernym/Hyponym edge

    G.add_edge(N[14], N[15], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[14], N[16], T_score=2, relation="syn_2")  # Synthesis edge
    G.add_edge(N[15], N[17], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[15], N[18], T_score=2, relation="syn_2")  # Synthesis edge
    G.add_edge(N[16], N[19], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[16], N[20], T_score=2, relation="syn_2")  # Synthesis edge
    G.add_edge(N[17], N[21], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[17], N[22], T_score=2, relation="syn_2")  # Synthesis edge
    G.add_edge(N[18], N[15.5], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[18], N[24], T_score=2, relation="syn_2")  # Synthesis edge

    G.add_edge(N[20], N[25], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[20], N[26], T_score=2, relation="syn_2")  # Synthesis edge
    G.add_edge(N[21], N[27], A_score=1, relation="syn_1")  # Antonym edge
    G.add_edge(N[21], N[28], T_score=2, relation="syn_2")  # Synthesis edge

    G.graph["branches"] = {
        N[1]: {
            (N[1], N[3]): (1.5, 1.4),
        },
        N[4]: {
            (N[4], N[6]): (1.6, 1.7),
        },
        N[7]: {
            (N[7], N[9]): (1.8, 1.9),
        },
        N[10]: {
            (N[10], N[12]): (1.10, 1.11),
        },
        N[14]: {
            (N[14], N[16]): (1.12, 1.13),
        },
        N[15]: {
            (N[15], N[17]): (1.14, 1.15),
        },
        N[16]: {
            (N[16], N[18]): (1.16, 1.17),
        },
    }
    return (
        N,
        N[21],
        {
            1: {
                1: {
                    1: {".": "plus"},
                    2: {".": "minus"},
                    3: {".": "plus minus n is 0"},
                    ".": "line calculation",
                },
                2: {
                    1: {".": "times"},
                    2: {".": "divided by"},
                    3: {".": "times divided n is 1"},
                    ".": "point calculation",
                },
                3: {
                    1: {".": "exponent"},
                    2: {".": "root"},
                    3: {".": "root of exponent is n"},
                    ".": "repeated calculation",
                },
                ".": "arithmetic",
            },
            2: {".": "number theory"},
            3: {".": "algebra"},
        },
    )


def test(scenario):
    # Add noise to the graph
    G = nx.MultiDiGraph()
    N, root, expectation = scenario(G)
    print("\n".join([f"{k}. {v}" for k, v in enumerate(N.values())]))
    for node, text in N.items():
        G.nodes[text]["text"] = text

    mst = construct_mst(G, root, 3)
    nested_result = serialize_graph_to_structure(mst, start_node=root, no_title=True)
    pprint(nested_result)
    try:
        assert nested_result == expectation
    except AssertionError:
        print("FAILED")
        pprint(dict_diff(expectation, nested_result))
        # pprint(expectation)
        pprint(nested_result)


if __name__ == "__main__":
    from integrator.test_mst import from_texts

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
        epochs=6,
        start_node="plus and minus",
        start_with_sub=True,
    )
    t.dump_graph("calc", t.graph, "calc")
    draw_graph(g, other_attributes="A_score")

    while True:
        with catchtime("test"):
            test(scenario_1)

            test(scenario_2)

            test(scenario_calc)

            break
