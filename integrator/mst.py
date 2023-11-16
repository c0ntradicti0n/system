from pprint import pprint

import networkx as nx

from integrator.serialize import serialize_graph_to_structure


def edge_priority(edge, edge_type):
    if edge_type == "ant":
        return edge[2].get("a_score", 0)
    elif edge_type == "syn":
        return edge[2].get("s_score", 0)
    elif edge_type == "sub":
        return edge[2].get("h_score", 0)
    return 0


def select_edges(G, node, visited, depth):
    selected_edges = []
    trident_pairs = {}

    # Alternating between ant-syn branchings and sub-segments
    if depth % 2 == 0:
        # Collecting ant and syn edges with trident attribute
        for n1, n2, attr in G.out_edges(node, data=True):
            if "trident" in attr:
                trident_id = attr["trident"]
                trident_pairs.setdefault(trident_id, []).append((n1, n2, attr))

        # Selecting the best ant-syn pair based on scores
        for pair in trident_pairs.values():
            if len(pair) == 2:
                ant_edge, syn_edge = pair
                if ant_edge[1] not in visited and syn_edge[1] not in visited:
                    selected_edges.extend(pair)
    else:
        sub_edges = [
            (n1, n2, attr)
            for n1, n2, attr in G.out_edges(node, data=True)
            if attr.get("relation") == "sub" and n2 not in visited
        ]
        if sub_edges:
            # Sort sub_edges for deterministic selection
            sub_edges.sort(key=lambda e: (-edge_priority(e, "sub"), e[1]))
            best_sub_edge = sub_edges[0]
            selected_edges.append(best_sub_edge)

    return selected_edges


def construct_mst(G, root, max_depth):
    visited = set()
    mst = []

    def dfs(node, depth):
        if depth > max_depth:
            return

        visited.add(node)
        selected_edges = select_edges(G, node, visited, depth)

        for _, child, attr in selected_edges:
            if child not in visited:
                mst.append((node, child, attr))
                dfs(child, depth + 1)

    dfs(root, 0)

    temp_subgraph = nx.DiGraph()
    for n1, n2, attr in mst:
        temp_subgraph.add_edge(n1, n2, **attr)

    # transfer alls attributes from G to temp_subgraph
    for node in temp_subgraph.nodes:
        temp_subgraph.nodes[node].update(G.nodes[node])

    return temp_subgraph


if __name__ == "__main__":
    # Create a MultiDiGraph
    G = nx.MultiDiGraph()

    import random

    # Create a MultiDiGraph
    G = nx.MultiDiGraph()

    # Add nodes with labels
    N = {
        1: "Be",
        2: "Not to be",
        3: "Becoming",
        4: "Same and not the Same of Being and Non-Being",
        5: "Coming-into-Being and Ceasing-to-Be",
        6: "Sublation",
    }

    # Add edges with attributes
    G.add_edge(N[1], N[2], a_score=1, relation="ant", trident=1)  # Antonym edge
    G.add_edge(N[4], N[5], a_score=1, relation="ant", trident=1)  # Antonym edge

    G.add_edge(N[1], N[3], s_score=0.8, relation="syn", trident=1)  # Synthesis edge
    G.add_edge(N[4], N[6], s_score=0.8, relation="syn", trident=1)  # Synthesis edge

    G.add_edge(N[3], N[4], h_score=0.5, relation="sub")  # Hypernym/Hyponym edge

    # update text attribute of nodes
    for node, text in N.items():
        G.nodes[text]["text"] = text

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
                    relation=relation,
                    trident=trident,
                )

            i += 1

    # Add noise to the graph
    add_noise_to_graph(G, edge_count=20)

    # Run the MST construction
    root = N[1]  # Starting node
    max_depth = 5  # Maximum depth for the DFS
    mst = construct_mst(G, root, max_depth)

    nested_result = serialize_graph_to_structure(mst, start_node=root)

    # Expected result (based on your description)
    expected_result = {
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

    pprint(nested_result)

    # Test if the result matches the expected structure
    print("Test Passed" if nested_result == expected_result else "Test Failed")
