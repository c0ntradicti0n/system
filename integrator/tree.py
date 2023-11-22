# cython: language_level=3
import itertools
import math
import os
import pickle
from collections import defaultdict
from pprint import pprint

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from integrator.mst import construct_mst
from lib.shape import view_shape

image_folder = "images/"
pickle_folder = "states/"


class Tree:
    def __init__(self, enumerated_texts):
        self.graph = nx.MultiDiGraph()
        self.j = 0
        self.inputs = enumerated_texts
        self._populate_graph(enumerated_texts)
        self._create_sequence_edges()
        self.params = defaultdict(lambda: None)

    def _populate_graph(self, enumerated_texts):
        for key, text in enumerated_texts:
            self.graph.add_node(key, text=text)

    def _create_sequence_edges(self):
        sorted_nodes = sorted(self.graph.nodes())
        for i in range(len(sorted_nodes) - 1):
            self.graph.add_edge(
                sorted_nodes[i], sorted_nodes[i + 1], relation="text_sequence"
            )

    def add_unique_relation(self, source, target, relation_type, **attributes):
        """
        Add or update an edge in a MultiDiGraph with unique 'relation' attribute for each outgoing edge.

        Parameters:
        G (nx.MultiDiGraph): The graph to which the edge will be added.
        source (node): The node from which the edge starts.
        target (node): The node to which the edge connects.
        **attributes: Edge attributes.
        """
        if relation_type is not None:
            # Check for an edge with the same 'relation' attribute
            for u, v, key, data in self.graph.edges(source, data=True, keys=True):
                if v == target and data.get("relation") == relation_type:
                    # Update the existing edge with new attributes
                    self.graph[source][target][key].update(attributes)
                    return 0

        # Add a new edge if no matching edge was found
        attributes["relation"] = relation_type

        self.graph.add_edge(source, target, **attributes)
        return 1

    def get_relation(self, source, target, relation_type):
        for u, v, key, data in self.graph.out_edges(source, data=True, keys=True):
            if target == v and data.get("relation") == relation_type:
                return data
        return None

    def add_relation(self, key1, key2, relation_type, **kwargs):
        if key1 in self.graph and key2 in self.graph:
            self.graph.add_edge(key1, key2, relation=relation_type, **kwargs)
        else:
            print(f"Nodes {key1} or {key2} not found in the graph.")

    def filter_edges(self, relation_type):
        for u, v, data in list(self.graph.edges(data=True)):
            if data.get("relation") == relation_type:
                yield (u, v)

    def remove_edge(self, key1, key2):
        if self.graph.has_edge(key1, key2):
            self.graph.remove_edge(key1, key2)

    @staticmethod
    def has_edge_with_attribute(G, source, target, attr_name, attr_value):
        """
        Check if there is an edge between source and target with the specified attribute and value.

        :param G: NetworkX graph
        :param source: source node
        :param target: target node
        :param attr_name: attribute name
        :param attr_value: attribute value
        :return: True if such an edge exists, False otherwise
        """
        if G.is_multigraph():
            # Check if the edge exists
            if (source, target) not in G.edges():
                return False
            # Check each edge for the attribute
            for key in G[source][target]:
                if G[source][target][key].get(attr_name) == attr_value:
                    return True
        else:
            # Check if the edge exists and has the attribute
            if (source, target) in G.edges() and G[source][target].get(
                attr_name
            ) == attr_value:
                return True
        return False

    @staticmethod
    def graph_without_text_sequence(graph=None):
        # Create a new DiGraph without "text_sequence" edges
        filtered_graph = nx.MultiDiGraph()
        for node in graph.nodes:
            for u, v, data in graph.out_edges(node, data=True):
                if data.get("relation") != "text_sequence":
                    filtered_graph.add_edge(u, v, **data)

        return filtered_graph

    def draw_graph(self, graph=None, root=None, path="graph.png", text_relation=True):
        if os.environ.get("NO_IMAGES", False):
            return
        try:
            import matplotlib.pyplot as plt

            if graph == None:
                graph = self.graph
            plt.figure(figsize=(10, 10), dpi=100)  # Adjust as needed
            plt.clf()  # Clear the current figure

            # pos = nx.spring_layout(graph)

            if isinstance(graph, nx.DiGraph):
                pos = graphviz_layout(graph, root=root, prog="twopi")
            else:
                pos = graphviz_layout(graph, root=root, prog="twopi")
            # pos = nx.circular_layout(self.graph)

            # color-code the edges
            color_code = {
                "the": "green",
                "ant": "red",
                "syn": "blue",
                "hie": "orange",
                **({"text_sequence": "gray"} if text_relation is True else {}),
            }
            edge_color_list = [
                color_code[rel[2]] for rel in graph.edges.data("relation")
            ]

            # Color nodes based on the 'start' attribute
            node_colors = [
                "yellow" if graph.nodes[node].get("start", False) else "1"
                for node in graph.nodes()
            ]

            nx.draw(
                graph,
                pos,
                with_labels=True,
                edge_color=edge_color_list,
                node_color=node_colors,
            )
            # edge_labels = nx.get_edge_attributes(graph, "relation")

            leg = plt.legend(color_code, labelcolor=color_code.values())
            for i, item in enumerate(leg.legendHandles):
                item.set_color(list(color_code.values())[i])

            plt.savefig(image_folder + path, format="png", bbox_inches="tight")
            plt.close()
        except:
            print("Error in drawing graph")
            pass

    def pull(self, n, _score=None, on_relation=None, for_relation=None):
        result = []
        while len(result) != n:
            for nn in self.nodes_with_least_info(
                self.graph,
                n,
                _score=_score,
                on_relation=on_relation,
                for_relation=for_relation,
            ):
                if len(result) < n:
                    if isinstance(nn, list):
                        for nnn in nn:
                            result.append((nnn, self.graph.nodes[nnn]["text"]))
                    else:
                        result.append((nn, self.graph.nodes[nn]["text"]))
                else:
                    break
        return result

    def pull_lz(self, n, m, _score=None, on_relation=None, for_relation=None):
        return view_shape(
            list(
                zip(
                    *self.pull(
                        n * m,
                        _score,
                        on_relation=on_relation,
                        for_relation=for_relation,
                    )
                )
            ),
            (2, n, m),
        )

    yielded = []

    def nodes_with_least_info(
        self, G, n, _score=None, on_relation=None, for_relation=None
    ):
        # Dictionary to store the sum of "_score" attributes for each node
        all_nodes = list(G.nodes())

        if for_relation:
            results = []
            for node1 in all_nodes:
                for node2 in all_nodes:
                    if node1 == node2:
                        continue
                    if not self.has_edge_with_attribute(
                        G, node1, node2, for_relation, "relation"
                    ):
                        results.append([node1, node2])
            return results

        node_scores = {}
        random.shuffle(all_nodes)
        for node in all_nodes:
            total_score = 0
            for _, _, data in G.edges(node, data=True):
                if on_relation and not any(
                    u
                    for u, v, d in G.in_edges(node, data=True)
                    if d["relation"] == on_relation
                ):
                    continue

                for key, value in data.items():
                    if key.endswith("_score") and (
                        _score and key.startswith(_score) or not _score
                    ):
                        total_score += value
            node_scores[node] = total_score

        # Sort nodes based on their total_score
        sorted_nodes = sorted(node_scores, key=node_scores.get)
        if len(sorted_nodes) < n:
            return sorted_nodes[:n]

        # Return the top n nodes with the least information
        return sorted_nodes[:n]

    @staticmethod
    def computed_out_edges(G, v):
        return list(Tree.computed_tri_out_edges(G, v)) + list(
            Tree.computed_sub_out_edges(G, v)
        )

    @staticmethod
    def computed_tri_out_edges(G, v):
        return [
            (n1, n2, attr)
            for n1, n2, attr in G.out_edges(v, data=True)
            if "trident" in attr
        ]

    @staticmethod
    def computed_sub_out_edges(G, v):
        return [
            (n1, n2, attr)
            for n1, n2, attr in G.out_edges(v, data=True)
            if attr["relation"] == "hie"
        ]

    def missing_edges(self, relation):
        graph_with_only_relation = self.graph_without_relation(self.graph, relation)
        result = [
            (n1, n2)
            for n1, n2 in itertools.permutations(graph_with_only_relation.nodes, 2)
            if not graph_with_only_relation.has_edge(n1, n2)
        ]

        return result

    def graph_without_relation(self, graph, relation):
        filtered_graph = nx.DiGraph()
        # add all nodes
        filtered_graph.add_nodes_from(graph.nodes(data=True))

        for node in graph.nodes:
            for u, v, data in graph.out_edges(node, data=True):
                if data.get("relation") == relation:
                    filtered_graph.add_edge(u, v, **data)

        return filtered_graph

    @staticmethod
    def computed_i_tri_out_edges(G, v, visited):
        tou = Tree.computed_tri_out_edges(G, v)
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
        return result

    @staticmethod
    def edge_score(e, _score=[]):
        total_score = 0
        for k, v in e.items():
            if v is None:  # Skip if the score is None
                continue
            if k.endswith("_score"):
                if not any(k.startswith(s) for s in _score):
                    continue
                total_score += v
        return total_score

    @staticmethod
    def node_score(G, node, _score=None):
        return sum(
            Tree.edge_score(attr, _score=_score)
            for _, _, attr in Tree.computed_out_edges(G, node)
        )

    @staticmethod
    def count_downstream_edges(G, node, visited=None):
        if visited is None:
            visited = set()

        # Mark the current node as visited
        visited.add(node)

        # Get all out edges of the current node
        out_edges = G.out_edges(node)

        # Count the number of out edges
        count = len(out_edges)

        # Recursively count downstream edges for each neighboring node
        for _, neighbor in out_edges:
            if neighbor not in visited:
                count += Tree.count_downstream_edges(G, neighbor, visited)

        return count

    @staticmethod
    def grow_subgraph(G, node, visited, depth, is_tri=False, last_subnode=None):
        if depth < 0:
            return [], []

        selected_edges = []

        # If no branching found, return
        if not is_tri:
            # Get ant-syn branching

            all_tri_out_edges = Tree.computed_i_tri_out_edges(G, node, visited)
            all_tri_out_edges = list(
                sorted(all_tri_out_edges, key=lambda x: x[0][2]["trident"])
            )
            if not all_tri_out_edges:
                return [], []

            tri_out_edges = max(
                all_tri_out_edges,
                key=lambda x: Tree.edge_score(x[0][2]) + Tree.edge_score(x[1][2]),
            )

            # Extract ant and syn nodes
            _, ant_node, ant_attr = tri_out_edges[0]
            _, syn_node, syn_attr = tri_out_edges[1]

            selected_edges.append((node, ant_node, ant_attr))
            selected_edges.append((node, syn_node, syn_attr))

            # Mark nodes as visited
            visited.add(ant_node)
            visited.add(syn_node)

            # Recursively grow subgraph for ant and syn branches
            ant_nodes, ant_edges = Tree.grow_subgraph(
                G, ant_node, visited, depth - 1, is_tri=True
            )
            syn_nodes, syn_edges = Tree.grow_subgraph(
                G, syn_node, visited, depth - 1, is_tri=True
            )
            the_nodes, the_edges = Tree.grow_subgraph(
                G, node, visited, depth - 1, is_tri=True
            )

            selected_edges.extend(the_edges)
            selected_edges.extend(ant_edges)
            selected_edges.extend(syn_edges)

            return [
                node,
                ant_node,
                syn_node,
                the_nodes,
            ] + ant_nodes + syn_nodes + the_nodes, selected_edges
        else:
            # Identify sub related nodes
            sub_out_edges = Tree.computed_sub_out_edges(G, node)
            sub_nodes = [n2 for _, n2, _ in sub_out_edges if n2 not in visited]

            # If there are sub nodes, select the one with highest score
            if sub_nodes:
                sub_scores = {n: Tree.node_score(G, n, _score="h_") for n in sub_nodes}
                best_sub_node = max(sub_scores, key=sub_scores.get)
                best_sub_edge_attr = next(
                    attr for n1, n2, attr in sub_out_edges if n2 == best_sub_node
                )

                selected_edges.append((node, best_sub_node, best_sub_edge_attr))

                visited.add(best_sub_node)

                # Recursively grow subgraph for the selected sub node
                sub_nodes, sub_edges = Tree.grow_subgraph(
                    G,
                    best_sub_node,
                    visited,
                    depth - 1,
                    is_tri=False,
                    last_subnode=[best_sub_node, best_sub_edge_attr],
                )
                selected_nodes = [node, best_sub_node] + sub_nodes
                selected_edges.extend(sub_edges)
            else:
                selected_nodes = []

        return selected_nodes, selected_edges

    @staticmethod
    def compute_node_score(
        G, node, scores, subsumption_score_sum, outgoing_edge_count, visited=None
    ):
        # If score already computed, return it
        visited = set() if visited is None else visited

        if scores[node] is not None:
            return scores[node]

        if node in visited:
            return 0

        # Compute direct average subsumption score
        if outgoing_edge_count[node] != 0:  # Avoid division by zero
            scores[node] = subsumption_score_sum[node] / outgoing_edge_count[node]
        else:
            scores[node] = 0

        # Add scores of nodes that this node subsumes
        for _, m, data in G.out_edges(node, data=True):
            scores[node] += Tree.compute_node_score(
                G,
                m,
                scores,
                subsumption_score_sum,
                outgoing_edge_count,
                visited=visited | {node, m},
            )

        # Store the computed score in the graph and return
        G.nodes[node]["score"] = scores[node]
        return scores[node]

    @staticmethod
    def top_n_subsuming_nodes(G, n=10):
        # Initialization
        scores = {node: None for node in G.nodes()}
        subsumption_score_sum = {node: 0 for node in G.nodes()}
        outgoing_edge_count = {node: 0 for node in G.nodes()}

        # Compute sum of scores and edge count
        for node in G.nodes():
            for _, _, data in G.in_edges(node, data=True):
                if "h_score" in data and data["h_score"]:
                    subsumption_score_sum[node] += data["h_score"]
                    outgoing_edge_count[node] += 1
                if "a_score" in data and data["a_score"]:
                    subsumption_score_sum[node] += data["a_score"] / 1
                    outgoing_edge_count[node] += 1 / 3
                if "s_score" in data and data["s_score"]:
                    subsumption_score_sum[node] += data["s_score"] / 1
                    outgoing_edge_count[node] += 1 / 3

        # Compute recursive scores for each node
        for node in G.nodes():
            Tree.compute_node_score(
                G, node, scores, subsumption_score_sum, outgoing_edge_count
            )

        # Sort nodes by their scores in descending order
        sorted_nodes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Return top n nodes
        return sorted_nodes[:n]

    def max_score_triangle_subgraph(
        self, __G, return_start_node=False, start_with_sub=False
    ):
        _G = __G.copy()

        G = Tree.graph_without_text_sequence(_G)

        depth = self.params["depth"]
        if not depth:
            try:
                depth = math.log(len(G.nodes()), 3)
            except:
                depth = 0

        # Sort nodes by score and get the top 10
        start_node = self.params["startNode"]
        if not start_node:
            sorted_nodes = Tree.top_n_subsuming_nodes(G, n=4)
        else:
            sorted_nodes = [start_node]

        best_subgraph = None
        best_start_node = None
        max_edges = 0

        for node in sorted_nodes:
            visited = {node}

            # Grow subgraph recursively
            result_nodes, result_edges = Tree.grow_subgraph(
                G, node=node, visited=visited, depth=depth
            )

            # Create a temporary subgraph for this start_node
            temp_subgraph = nx.DiGraph()
            for n1, n2, attr in result_edges:
                temp_subgraph.add_edge(n1, n2, **attr)
            # If this subgraph has more edges than the best one found so far, update best_subgraph
            if len(temp_subgraph.edges) > max_edges:
                max_edges = len(temp_subgraph.edges)
                best_subgraph = temp_subgraph
                best_start_node = node

        if best_subgraph:
            for node in best_subgraph.nodes:
                best_subgraph.nodes[node].update(_G.nodes[node])
            # If the best subgraph's start_node exists in the subgraph, mark it

            best_subgraph.nodes[best_start_node]["start"] = True
        if return_start_node:
            return best_subgraph or nx.DiGraph(), best_start_node

        return best_subgraph or nx.DiGraph()

    def save_state(self, number, hash, filename=None):
        os.makedirs(self.pickle_folder_path(hash), exist_ok=True)
        if filename is None:
            filename = os.path.join(
                self.pickle_folder_path(hash), f"tree_state_{number}.pkl"
            )
        state = {
            "inputs": self.inputs,
            "graph": self.graph,
            "yielded": self.yielded,
            "j": self.j,
            "params": dict(self.params),
        }
        with open(filename, "wb") as file:
            pickle.dump(state, file)

    @classmethod
    def pickle_folder_path(cls, hash=None):
        return os.path.join(pickle_folder, hash if hash else "")

    @classmethod
    def load_state(cls, hash, number=None):
        os.makedirs(cls.pickle_folder_path(hash), exist_ok=True)
        files = [
            f
            for f in os.listdir(cls.pickle_folder_path(hash))
            if f.startswith(f"tree_state") and f.endswith(".pkl")
        ]
        if not files:
            return None, None

        if number is None:
            numbers = [int(f.split("_")[2].split(".")[0]) for f in files]
            latest_number = max(numbers)
            filename = os.path.join(
                cls.pickle_folder_path(hash), f"tree_state_{latest_number}.pkl"
            )
        else:
            filename = os.path.join(
                cls.pickle_folder_path(hash), f"tree_state_{number}.pkl"
            )
            latest_number = number

        try:
            with open(filename, "rb") as file:
                state = pickle.load(file)
                tree = cls([])
                tree.inputs = state["inputs"]
                tree.graph = state["graph"]
                tree.yielded = state["yielded"]
                tree.j = state["j"] + 1
                tree.load_params(hash)

                return tree, latest_number + 1
        except MemoryError:
            print("Memory error in loading state")

            try:
                filename = os.path.join(
                    cls.pickle_folder_path(hash), f"tree_state_{latest_number-1}.pkl"
                )
                with open(filename, "rb") as file:
                    state = pickle.load(file)
                    tree = cls([])
                    tree.inputs = state["inputs"]
                    tree.graph = state["graph"]
                    tree.yielded = state["yielded"]
                    tree.j = state["j"] + 1
                    tree.load_params(hash)

                    return tree, latest_number + 1
            except:
                print("Memory error in loading state")
                return None, None

    def load_params(self, hash):
        if os.path.exists(f"{self.pickle_folder_path()}/{hash}-params.pkl"):
            with open(f"{self.pickle_folder_path()}/{hash}-params.pkl", "rb") as f:
                params = pickle.load(f)
            self.params.update(params)
        else:
            self.params = defaultdict(lambda: None)

    def dump_graph(self, hash, graph=None, filename=None):
        if graph is None:
            graph = self.graph
        if filename is None:
            filename = os.path.join(self.pickle_folder_path(hash), f"tree_graph.txt")

        with open(filename, "wb") as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

    def get_text(self, node):
        return self.graph.nodes[node]["text"]


def test_max_score_triangle_subgraph():
    # Create a Tree instance with some nodes
    enumerated_texts = [(str(i), f"Text {i}") for i in range(12)]
    tree = Tree(enumerated_texts)

    edges_to_create = [
        ("2", "3", "ant", 1, {"v_score": 0.5}),
        ("2", "4", "syn", 1, {"v_score": 0.5}),
        ("3", "5", "hie", None, {"h_score": 0.4}),
        ("5", "6", "ant", 2, {"v_score": 0.4}),
        ("5", "7", "syn", 2, {"v_score": 0.4}),
        ("4", "8", "hie", None, {"h_score": 0.4}),
        ("8", "9", "ant", 3, {"v_score": 0.3}),
        ("8", "10", "syn", 3, {"v_score": 0.3}),
        ("10", "11", "hie", None, {"h_score": 0.3}),
    ]

    for n1, n2, rel, trident, attr in edges_to_create:
        tree.add_relation(n1, n2, relation_type=rel, trident=trident, **attr)

    # Get the max score path
    result_graph = tree.max_score_triangle_subgraph(tree.graph)

    print(result_graph.edges(data=True))
    for n1, n2, rel, trident, attr in edges_to_create:
        assert result_graph.has_edge(
            n1, n2
        ), f"Edge {n1, n2} not found in the result graph."
    pprint(Tree.serialize_graph_to_structure(result_graph, "2"))

    pprint("Test passed!")


def test_max_score_triangle_subgraph_worse_paths():
    # Create a Tree instance with some nodes
    enumerated_texts = [(str(i), f"Text {i}") for i in range(12)]
    tree = Tree(enumerated_texts)

    edges_to_create = [
        ("2", "3", "ant", 1, {"v_score": 0.5}),
        ("2", "4", "syn", 1, {"v_score": 0.5}),
        ("3", "5", "hie", None, {"h_score": 0.4}),
        ("5", "6", "ant", 2, {"v_score": 0.4}),
        ("5", "7", "syn", 2, {"v_score": 0.4}),
        ("4", "8", "hie", None, {"h_score": 0.4}),
        ("8", "9", "ant", 3, {"v_score": 0.3}),
        ("8", "10", "syn", 3, {"v_score": 0.3}),
        ("10", "11", "hie", None, {"h_score": 0.3}),
    ]

    for n1, n2, rel, trident, attr in edges_to_create:
        tree.add_relation(n1, n2, relation_type=rel, trident=trident, **attr)

    # Get the max score path
    result_graph = tree.max_score_triangle_subgraph(tree.graph)

    print(result_graph.edges(data=True))
    for n1, n2, rel, trident, attr in edges_to_create:
        assert result_graph.has_edge(
            n1, n2
        ), f"Edge {n1, n2} not found in the result graph."

    pprint("Test passed!")


import random


def generate_test_data(num_nodes=100, max_score=1.0):
    enumerated_texts = [(str(i), f"Node {i}") for i in range(1, num_nodes + 1)]
    tree = Tree(enumerated_texts)

    edges_to_create = []
    used_tridents = set()

    total_relations = num_nodes * 17
    created_relations = 0

    while created_relations < total_relations:
        i = random.randint(1, num_nodes)
        next_node = random.randint(1, num_nodes)
        while next_node == i:  # Ensure different nodes
            next_node = random.randint(1, num_nodes)

        relation_type = random.choice(["ant", "syn_1", "syn_2", "hie"])

        if relation_type in ["ant", "syn"]:
            trident = random.randint(1, num_nodes // 2)
            possible_tridents = [
                x for x in range(1, num_nodes // 2 + 1) if x not in used_tridents
            ]
            if possible_tridents:  # Only proceed if there are unused trident values
                trident = random.choice(possible_tridents)
                used_tridents.add(trident)
            else:  # No trident values left; skip this iteration and continue
                continue
            used_tridents.add(trident)

            v_score = random.uniform(0.1, max_score)
            h_score = None

            # Add ant edge
            edges_to_create.append(
                (
                    str(i),
                    str(next_node),
                    "ant",
                    trident,
                    {"v_score": v_score, "h_score": h_score},
                )
            )

            # Add corresponding syn edge
            possible_nodes = [x for x in range(1, num_nodes + 1) if x != next_node]
            next_node_syn = random.choice(possible_nodes)

            edges_to_create.append(
                (
                    str(i),
                    str(next_node_syn),
                    "syn",
                    trident,
                    {"v_score": v_score, "h_score": h_score},
                )
            )

            created_relations += 2

        else:
            v_score = None
            h_score = random.uniform(0.1, max_score)
            trident = None
            edges_to_create.append(
                (
                    str(i),
                    str(next_node),
                    relation_type,
                    trident,
                    {"v_score": v_score, "h_score": h_score},
                )
            )

            created_relations += 1

    for n1, n2, rel, trident, attr in edges_to_create:
        tree.add_relation(n1, n2, relation_type=rel, trident=trident, **attr)

    return tree


def test_with_generated_data():
    tree = generate_test_data()
    result_graph, start_node = tree.max_score_triangle_subgraph(
        tree.graph, return_start_node=True
    )
    tree.draw_graph(result_graph, path="generated_graph.png", text_relation=False)
    pprint(Tree.serialize_graph_to_structure(result_graph, start_node))

    # Assertion 1: Every "ant" has a corresponding "syn" with the same trident value and vice versa.
    ant_edges = [
        (u, v, d)
        for u, v, d in result_graph.edges(data=True)
        if d.get("relation") == "ant"
    ]
    syn_edges = [
        (u, v, d)
        for u, v, d in result_graph.edges(data=True)
        if d.get("relation") == "syn"
    ]
    for u, v, d in ant_edges:
        trident_value = d.get("trident")
        assert any(
            d_syn.get("trident") == trident_value for _, _, d_syn in syn_edges
        ), f"No matching 'syn' for 'ant' with trident value {trident_value}"
    for u, v, d in syn_edges:
        trident_value = d.get("trident")
        assert any(
            d_ant.get("trident") == trident_value for _, _, d_ant in ant_edges
        ), f"No matching 'ant' for 'syn' with trident value {trident_value}"

    # Assertion 2: No sub-sub -relations:
    for edge in result_graph.edges(data=True):
        source, target, attr = edge
        if attr["relation"] == "hie":
            # Flags to check if the intermediate node has both 'ant' and 'syn' edges
            has_ant = False
            has_syn = False
            for _, next_target, next_attr in result_graph.out_edges(target, data=True):
                if next_attr["relation"] == "ant":
                    has_ant = True
                elif next_attr["relation"] == "syn":
                    has_syn = True
                # If the next relation is 'sub' and the intermediate node does not have both 'ant' and 'syn' relations
                if next_attr["relation"] == "hie" and not (has_ant and has_syn):
                    raise AssertionError(
                        f"Found a sub-sub relation from {source} -> {target} -> {next_target} without both 'ant' and 'syn' relations."
                    )

    # Assertion 3: Validate the edge scores.
    for _, _, d in result_graph.edges(data=True):
        if d.get("relation") in ["ant", "syn"]:
            assert (
                0.1 <= d.get("v_score") <= 1.0
            ), f"Invalid 'v_score' value {d.get('v_score')}"
        if d.get("relation") == "hie":
            assert (
                0.1 <= d.get("h_score") <= 1.0
            ), f"Invalid 'h_score' value {d.get('h_score')}"

    print(result_graph.edges)
    print("Test with generated data passed!")


if __name__ == "__main__":
    test_max_score_triangle_subgraph()
    test_max_score_triangle_subgraph_worse_paths()
    test_with_generated_data()

    enumerated_texts = [
        ("2", "* \tDie Welt ist alles, was der Fall ist."),
        ("6", "* \tDie Welt ist alles, was der Fall ist."),
        ("5", "* \tDie Welt ist alles, was der Fall ist."),
        ("1.1", "Die Welt ist die Gesamtheit der Tatsachen, nicht der Dinge."),
    ]
    graph_wrapper = Tree(enumerated_texts)
    graph_wrapper.draw_graph()
