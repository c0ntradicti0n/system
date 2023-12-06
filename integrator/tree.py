import itertools
import logging
import math
import os
import pickle
from collections import defaultdict
from functools import reduce
from pprint import pprint

import _pickle
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout

from integrator.combination_state import CustomCombinations
from integrator.mst_maximax import (computed_i_tri_out_edges,
                                    computed_sub_out_edges, maximax)
from lib.g_from_numpy_array import from_numpy_array
from lib.max_islice import maxislice
from lib.shape import flatten, view_shape
from lib.t import catchtime

image_folder = "images/"
pickle_folder = "states/"


class Tree:
    def __init__(self, enumerated_texts):
        self.j = 0
        self.inputs = enumerated_texts
        self.relation_types = ["hie", "ant", "syn_1", "syn_2"]

        self.node_index = {}
        self.index_text = {}
        self.i_to_key = {}
        self._populate_graph(enumerated_texts)
        self.iterators = {}
        self.params = defaultdict(lambda: None)
        self.branches = defaultdict(lambda: dict())

        self.matrices = {}
        for rel in self.relation_types:
            self.matrices[rel] = np.zeros(
                (len(self.node_index), len(self.node_index)), dtype=float
            )

    def _populate_graph(self, enumerated_texts):
        for i, (key, text) in enumerate(enumerated_texts):
            if key not in self.node_index:
                self.node_index[key] = len(self.node_index)
                self.index_text[key] = text
                self.i_to_key[i] = key

    def add_relation(self, key1, key2, relation_type, score):
        if relation_type not in self.relation_types:
            raise ValueError(f"Unknown relation type: {relation_type}")

        if key1 in self.node_index and key2 in self.node_index:
            i, j = self.node_index[key1], self.node_index[key2]
            self.matrices[relation_type][i, j] = score
        else:
            print(f"Nodes {key1} or {key2} not found in the node index.")

    def add_branching(self, source, target_1, target_2, score_1, score_2):
        if not source in self.branches:
            self.branches[source] = dict()
        self.branches[source][(target_1, target_2)] = (score_1, score_2)

    def get_relation(self, source, target, relation):
        if source in self.node_index and target in self.node_index:
            source_idx = self.node_index[source]
            target_idx = self.node_index[target]
            if relation in self.matrices:
                return self.matrices[relation][source_idx, target_idx]
            else:
                print(f"Relation type '{relation}' not recognized.")
                return None
        else:
            print(f"Nodes {source} or {target} not found.")
            return None

    @property
    def graph(self):
        with catchtime("graph"):
            g1 = from_numpy_array(
                self.matrices["hie"],
                parallel_edges=False,
                create_using=nx.MultiDiGraph,
                attr_name="hie_score",
                additional_attrs={"relation": "hie"},
            )
            g2 = from_numpy_array(
                self.matrices["ant"],
                parallel_edges=False,
                create_using=nx.MultiDiGraph,
                attr_name="ant_score",
                additional_attrs={"relation": "ant"},
            )
            g3 = from_numpy_array(
                self.matrices["syn_1"],
                parallel_edges=False,
                create_using=nx.MultiDiGraph,
                attr_name="A_score",
                additional_attrs={"relation": "syn_1"},
            )
            g4 = from_numpy_array(
                self.matrices["syn_2"],
                parallel_edges=False,
                create_using=nx.MultiDiGraph,
                attr_name="T_score",
                additional_attrs={"relation": "syn_2"},
            )

            graph_list = [g1, g2, g3, g4]
            g = reduce(nx.compose, graph_list)

            for key, idx in self.node_index.items():
                g.add_node(key)
                # add text
                g.nodes[key]["text"] = self.index_text[key]

            g = nx.relabel_nodes(g, self.i_to_key)

            g.graph["branches"] = dict(self.branches)
            return g

    @staticmethod
    def graph_without_text_sequence(graph=None):
        # Create a new DiGraph without "text_sequence" edges
        filtered_graph = nx.MultiDiGraph()
        filtered_graph.graph = graph.graph
        for node in graph.nodes:
            for u, v, data in graph.out_edges(node, data=True):
                if data.get("relation") != "text_sequence":
                    filtered_graph.add_edge(u, v, **data)

        return filtered_graph

    def draw_graph(self, graph=None, root=None, path="graph.png", text_relation=True):
        if os.environ.get("NO_IMAGES", False):
            return
        try:
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            import matplotlib.pyplot as plt

            if graph == None:
                graph = self.graph
            plt.figure(figsize=(10, 10), dpi=100)  # Adjust as needed
            plt.clf()  # Clear the current figure

            # pos = nx.spring_layout(graph)

            try:
                if isinstance(graph, nx.DiGraph):
                    pos = graphviz_layout(graph, root=root, prog="twopi")
                else:
                    pos = graphviz_layout(graph, root=root, prog="twopi")
            except Exception as e:
                pos = nx.circular_layout(graph)

            # color-code the edges
            color_code = {
                "syn_1": "red",
                "ant": "orange",
                "syn_2": "magenta",
                "hie": "blue",
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

            leg = plt.legend(color_code, labelcolor=color_code.values())
            for i, item in enumerate(leg.legendHandles):
                item.set_color(list(color_code.values())[i])

            plt.savefig(image_folder + path, format="png", bbox_inches="tight")
            plt.close()
        except Exception as e:
            logging.error(f"Error in drawing graph {e}", exc_info=True)

    def pull(self, n_samples, relations, on=None, on_indices =None):
        if not tuple(relations) in self.iterators:
            self.iterators[tuple(relations)] = CustomCombinations(
                list(self.node_index.keys()), n_samples, on=on, on_indices=on_indices
            )
        for k in self.iterators[tuple(relations)]:
            yield [(i, self.index_text[i]) for i in k]

    def pull_batch(self, batch_size, n_samples, relations=None, on=None ,on_indices =None):
        """
        :param batch_size:
        :param n_samples:
        :param relations:
        :return: keys, inputs
        """
        return view_shape(
            tuple(
                zip(
                    *[
                        ([i for i, _ in samples], [s for _, s in samples])
                        for samples in maxislice(
                            self.pull(n_samples, relations, on=on, on_indices=on_indices), batch_size
                        )
                    ]
                )
            ),
            (2, -1, n_samples),
        )

    def missing_edges(self, relation):
        graph_with_only_relation = self.graph_without_relation(self.graph, relation)
        result = (
            (n1, n2)
            for n1, n2 in itertools.permutations(graph_with_only_relation.nodes, 2)
            if not graph_with_only_relation.has_edge(n1, n2)
        )

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

            all_tri_out_edges = computed_i_tri_out_edges(G, node, visited)

            if not all_tri_out_edges:
                return [], []
            tri_out_edges = all_tri_out_edges[0]

            # Extract ant and syn nodes
            ant_node, score1 = tri_out_edges[0]
            syn_node, score2 = tri_out_edges[1]

            selected_edges.append(
                (node, ant_node, {"relation": "syn_1", "syn_1_score": score1})
            )
            selected_edges.append(
                (node, syn_node, {"relation": "syn_2", "syn_2_score": score2})
            )

            # Recursively grow subgraph for ant and syn branches
            ant_nodes, ant_edges = Tree.grow_subgraph(
                G, ant_node, visited, depth - 1, is_tri=True
            )
            ant_nodes = [n for n in ant_nodes if not n == []]
            try:
                visited.update(flatten(ant_nodes))
            except:
                raise
            syn_nodes, syn_edges = Tree.grow_subgraph(
                G, syn_node, visited, depth - 1, is_tri=True
            )
            syn_nodes = [n for n in syn_nodes if not n == []]

            visited.update(flatten(syn_nodes))

            the_nodes, the_edges = Tree.grow_subgraph(
                G, node, visited, depth - 1, is_tri=True
            )
            the_nodes = [n for n in the_nodes if not n == []]
            visited.update(flatten(the_nodes))

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
            sub_edges = computed_sub_out_edges(G, node, visited, [])

            # If there are sub nodes, select the one with highest score
            if sub_edges:
                if not sub_edges:
                    return [], []
                _, best_sub_node, best_sub_edge_attr = sub_edges[0]
                selected_edges.append((node, best_sub_node, best_sub_edge_attr))

                visited.update({n for e in selected_edges for n in e[:2]})

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
                if "hie_score" in data and data["hie_score"]:
                    subsumption_score_sum[node] += data["hie_score"]
                    outgoing_edge_count[node] += 1
                if "A_score" in data and data["A_score"]:
                    subsumption_score_sum[node] += data["A_score"] / 1
                    outgoing_edge_count[node] += 1 / 3
                if "T_score" in data and data["T_score"]:
                    subsumption_score_sum[node] += data["T_score"] / 1
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
        self, __G, return_start_node=False, start_with_sub=False, start_node=None
    ):
        _G = __G.copy()

        G = Tree.graph_without_text_sequence(_G)

        depth = self.params["depth"]
        if not depth:
            try:
                depth = math.log(len(G.nodes()), 3)
            except:
                depth = 0

        if depth > 4:
            depth = 4

        # Sort nodes by score and get the top 10
        start_node = start_node if start_node else self.params["startNode"]
        if not start_node:
            with catchtime("top_n_subsuming_nodes"):
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
                try:
                    best_subgraph.nodes[node].update(_G.nodes[node])
                except:
                    logging.error(f"Error in updating node {node}", exc_info=True)
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
            "matrices": self.matrices,
            "params": dict(self.params),
            "iterators": self.iterators,
            "node_index": self.node_index,
            "index_text": self.index_text,
            "branches": dict(self.branches),
            "j": self.j,
            "i_to_key": self.i_to_key,
        }
        with catchtime("pickle"):
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
                tree.matrices = state["matrices"]
                tree.j = state["j"] + 1
                tree.iterators = state["iterators"]
                tree.node_index = state["node_index"]
                tree.index_text = state["index_text"]
                tree.i_to_key = state["i_to_key"]
                tree.branches = defaultdict(None)
                if "branches" in state:
                    tree.branches.update(state["branches"])
                tree.load_params(hash)

                return tree, latest_number + 1
        except MemoryError:
            logging.error("Memory error in loading state", exc_info=True)
            os.unlink(filename)
            return Tree.load_state(hash, latest_number - 1)
        except KeyError:
            logging.error("Key error in loading state", exc_info=True)
            os.unlink(filename)
            return Tree.load_state(hash, latest_number - 1)
        except EOFError:
            logging.error("EOF error in loading state", exc_info=True)
            os.unlink(filename)
            return Tree.load_state(hash, latest_number - 1)
        except _pickle.UnpicklingError as e:
            logging.error(
                f"Unpickling error in loading state {filename}", exc_info=True
            )
            os.unlink(filename)
            return Tree.load_state(hash, latest_number - 1)
        except FileNotFoundError:
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
        return self.index_text[node]

    def all_computed(self, relations):
        if not tuple(relations) in self.iterators:
            return False
        return self.iterators[tuple(relations)].is_exhausted()

    def finished(self):
        return len(self.iterators) == 3 and all(
            iterator.is_exhausted() for iterator in self.iterators.values()
        )

    def progress(self):
        percentages = {
            str("_".join(kind)): iterator.get_percentage()
            for kind, iterator in self.iterators.items()
        }
        return percentages

    def get_all_texts(self):
        return self.inputs


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
                # If the next relation is 'hie' and the intermediate node does not have both 'ant' and 'syn' relations
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
