import copy
import pickle
import random
from pprint import pprint
import networkx as nx

from lib.helper import e, t
from integrator.mst import edge_priority
from integrator.serialize import serialize_graph_to_structure
from lib.t import catchtime, indented


def computed_out_edges(G, v):
    return list(computed_tri_out_edges(G, v)) + list(computed_sub_out_edges(G, v))


def computed_tri_out_edges(G, v):
    return [
        (n1, n2, attr)
        for n1, n2, attr in G.out_edges(v, data=True)
        if "trident" in attr
    ]


def find_sub_edge_score(G, u, v):
    for n1, n2, attr in G.out_edges(u, data=True):
        if n2 == v and attr["relation"] == "sub":
            return attr["h_score"]
    return 0.3


def computed_sub_out_edges(G, v, visited, parent):
    return list(
        sorted(
            [
                (
                    n1,
                    n2,
                    {
                        **attr,
                        "h_score": attr["h_score"] + find_sub_edge_score(G, parent, n2),
                    },
                )
                for n1, n2, attr in G.out_edges(v, data=True)
                if attr["relation"] == "sub" and n2 not in visited
            ],
            key=lambda x: x[2]["h_score"],
            reverse=True,
        )
    )


def computed_i_tri_out_edges(G, v, visited):
    tou = computed_tri_out_edges(G, v)
    if not tou:
        return []
    i_s = set([attr["trident"] for n1, n2, attr in tou])
    result = []
    for i in i_s:
        gs = [
            (n1, n2, attr)
            for n1, n2, attr in tou
            if attr["trident"] == i and n2 not in visited
        ]
        if len(gs) != 2:  # might be in case we visited that yet for this branch
            continue
        gs = sorted(gs, key=lambda x: x[2]["relation"])
        result.append(tuple(gs))
    result = sorted(
        result, key=lambda x: x[0][2].get("a_score", 0) + x[1][2].get("s_score",0), reverse=True
    )
    return result


def determine_edge_type(edge):
    # Assuming edge attributes include 'relation'
    relation = edge[2].get("relation", "")
    if relation == "ant":
        return "ant"
    elif relation == "syn":
        return "syn"
    elif relation == "sub":
        return "sub"
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


def maximax(node, depth, is_maximizing_player, G, visited, current_mst, parent=None):
    depth = 3
    def finale():
        return score_of_mst(current_mst), current_mst

    if depth == 0 or all_neighbors_visited(node, visited, G):
        return finale()
    best_mst = None
    max_eval = float("-inf")

    original_mst = copy.deepcopy(current_mst)
    original_visited = copy.deepcopy(visited)

    if is_maximizing_player:
        sub_edges = computed_sub_out_edges(G, node, visited, parent=parent)
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
            node, child_a, attr_a = edge_a
            node, child_b, attr_b = edge_b
            if child_a == child_b:
                print("same child")
                continue

            visited = copy.deepcopy(original_visited)
            current_mst = copy.deepcopy(original_mst)

            if child_a not in visited and child_b not in visited:
                visited.add(child_a)
                visited.add(child_b)
                visited.add(node)

                current_mst.add_edge(node, child_a, **attr_a)
                current_mst.add_edge(node, child_b, **attr_b)

                eval_b, mst_b = maximax(
                    child_b,
                    depth - 1,
                    True,
                    G,
                    visited,
                    copy.deepcopy(current_mst),
                    parent=node,
                )

                eval_a, mst_a = maximax(
                    child_a,
                    depth - 1,
                    True,
                    G,
                    set(mst_b.nodes),
                    copy.deepcopy(mst_b),
                    parent=node,
                )

                eval_0, mst_0 = maximax(
                    node,
                    depth - 1,
                    True,
                    G,
                    set(mst_a.nodes),
                    copy.deepcopy(mst_a),
                    parent=node,
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

        # Run the maximax algorithm
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
    return {i: n["text"] for i, n in G.nodes(data=True)}, "linear operations", {}


def scenario_1(G):
    N = {
        1: "Be",
        2: "Not to be",
        3: "Becoming",
        4: "Same and not the Same of Being and Non-Being",
        5: "Coming-into-Being and Ceasing-to-Be",
        6: "Sublation",
    }
    G.add_edge(N[1], N[2], a_score=1, relation="ant", trident=1)  # Antonym edge
    G.add_edge(N[4], N[5], a_score=1.1, relation="ant", trident=1)  # Antonym edge
    G.add_edge(N[1], N[3], s_score=2, relation="syn", trident=1)  # Synthesis edge
    G.add_edge(N[4], N[6], s_score=2.1, relation="syn", trident=1)  # Synthesis edge
    G.add_edge(N[3], N[4], h_score=3, relation="sub")  # Hypernym/Hyponym edge

    expectation = {
        1: {".": "[Be] Be"},
        2: {".": "[Not to be] Not to be"},
        3: {
            1: {
                ".": "[Same and not the Same of Being and Non-Being] Same and not the "
                "Same of Being and Non-Being"
            },
            2: {
                ".": "[Coming-into-Being and Ceasing-to-Be] Coming-into-Being and "
                "Ceasing-to-Be"
            },
            3: {".": "[Sublation] Sublation"},
            ".": "[Becoming] Becoming",
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

    G.add_edge(N[0], N[21], h_score=30, relation="sub")  # Hypernym/Hyponym edge
    G.add_edge(N[21], N[20], a_score=10, relation="ant", trident=0)  # Antonym edge
    G.add_edge(N[21], N[22], s_score=20, relation="syn", trident=0)  # Synthesis edge

    G.add_edge(N[21], N[10], h_score=30, relation="sub")  # Hypernym/Hyponym edge

    G.add_edge(N[1], N[2], a_score=1, relation="ant", trident=1)  # Antonym edge
    G.add_edge(N[1], N[3], s_score=2, relation="syn", trident=1)  # Synthesis edge
    G.add_edge(N[4], N[5], a_score=1, relation="ant", trident=2)  # Antonym edge
    G.add_edge(N[4], N[6], s_score=2, relation="syn", trident=2)  # Synthesis edge
    G.add_edge(N[7], N[8], a_score=1, relation="ant", trident=3)  # Antonym edge
    G.add_edge(N[7], N[9], s_score=2, relation="syn", trident=3)  # Synthesis edge
    G.add_edge(N[10], N[11], a_score=1, relation="ant", trident=4)  # Antonym edge
    G.add_edge(N[10], N[12], s_score=2, relation="syn", trident=4)  # Synthesis edge
    G.add_edge(N[10], N[1], h_score=3, relation="sub")  # Hypernym/Hyponym edge
    G.add_edge(N[11], N[4], h_score=3, relation="sub")  # Hypernym/Hyponym edge
    G.add_edge(N[12], N[7], h_score=3, relation="sub")  # Hypernym/Hyponym edge

    G.add_edge(N[14], N[15], a_score=1, relation="ant", trident=5)  # Antonym edge
    G.add_edge(N[14], N[16], s_score=2, relation="syn", trident=5)  # Synthesis edge
    G.add_edge(N[15], N[17], a_score=1, relation="ant", trident=6)  # Antonym edge
    G.add_edge(N[15], N[18], s_score=2, relation="syn", trident=6)  # Synthesis edge
    G.add_edge(N[16], N[19], a_score=1, relation="ant", trident=7)  # Antonym edge
    G.add_edge(N[16], N[20], s_score=2, relation="syn", trident=7)  # Synthesis edge
    G.add_edge(N[17], N[21], a_score=1, relation="ant", trident=8)  # Antonym edge
    G.add_edge(N[17], N[22], s_score=2, relation="syn", trident=8)  # Synthesis edge
    G.add_edge(N[18], N[15.5], a_score=1, relation="ant", trident=9)  # Antonym edge
    G.add_edge(N[18], N[24], s_score=2, relation="syn", trident=9)  # Synthesis edge

    G.add_edge(N[20], N[25], a_score=1, relation="ant", trident=10)  # Antonym edge
    G.add_edge(N[20], N[26], s_score=2, relation="syn", trident=10)  # Synthesis edge
    G.add_edge(N[21], N[27], a_score=1, relation="ant", trident=11)  # Antonym edge
    G.add_edge(N[21], N[28], s_score=2, relation="syn", trident=11)  # Synthesis edge

    return (
        N,
        N[0],
        {
            1: {
                1: {".": "[plus] plus"},
                2: {".": "[minus] minus"},
                3: {".": "[plus minus n is 0] plus minus n is 0"},
                ".": "[line calculation] line calculation",
            },
            2: {
                1: {".": "[times] times"},
                2: {".": "[divided by] divided by"},
                3: {".": "[times divided n is 1] times divided n is 1"},
                ".": "[point calculation] point calculation",
            },
            3: {
                1: {".": "[exponent] exponent"},
                2: {".": "[root] root"},
                3: {".": "[root of exponent is n] root of exponent is n"},
                ".": "[repeated calculation] repeated calculation",
            },
        },
    )


def test(scenario):
    # Function to add noise in the form of random edges and weights
    def add_noise_to_graph(G, edge_count=10):
        i = 100
        nodes = list(G.nodes)
        for _ in range(edge_count):
            n1, n2, n3 = random.sample(nodes, 3)

            a_score = -random.random()
            s_score = -random.random()
            h_score = -random.random()
            relation = random.choice(["ant", "syn", "sub"])

            if relation in ["ant", "syn"]:
                if G.has_edge(n1, n2):
                    co_edges = G[n1][n2]
                    if any([e["relation"] == relation for e in co_edges.values()]):
                        continue
                if G.has_edge(n1, n3):
                    co_edges2 = G[n1][n3]
                    if any([e["relation"] == relation for e in co_edges2.values()]):
                        continue

            trident = i if relation in ["ant", "syn"] else None
            G.add_edge(
                n1,
                n2,
                a_score=a_score,
                s_score=s_score,
                h_score=h_score,
                relation=relation,
                trident=trident,
            )
            if relation in ["ant", "syn"]:
                G.add_edge(
                    n1,
                    n3,
                    a_score=a_score,
                    s_score=s_score,
                    h_score=h_score,
                    relation={"ant": "syn", "syn": "ant"}[relation],
                    trident=trident,
                )

            i += 1

    # Add noise to the graph
    G = nx.MultiDiGraph()
    N, root, expectation = scenario(G)
    print ("\n".join([f"{k}. {v}" for k, v in enumerate(N.values())]))
    for node, text in N.items():
        G.nodes[text]["text"] = text
    # add_noise_to_graph(G, edge_count=20)

    mst = construct_mst(G, root, 3)
    nested_result = serialize_graph_to_structure(mst, start_node=root, no_title=True)
    pprint(nested_result)
    with t:
        assert nested_result == expectation


if __name__ == "__main__":
    while True:
        with catchtime("test"):
            test(scenario_calc)
            test(scenario_1)
            test(scenario_2)
            break
