import os
import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from lib.shape import view_shape

image_folder = "images/"
pickle_folder = "textnets/"

class Tree:
    def __init__(self, enumerated_texts):
        self.graph = nx.MultiDiGraph()
        self.inputs = enumerated_texts
        self._populate_graph(enumerated_texts)
        self._create_sequence_edges()

    def _populate_graph(self, enumerated_texts):
        for key, text in enumerated_texts:
            self.graph.add_node(key, text=text)

    def _create_sequence_edges(self):
        sorted_nodes = sorted(self.graph.nodes())
        for i in range(len(sorted_nodes) - 1):
            self.graph.add_edge(
                sorted_nodes[i], sorted_nodes[i + 1], relation="text_sequence"
            )

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



    def draw_graph_without_text_sequence(self, graph=None, root=None, path="graph.png"):
        if graph is None:
            graph = self.graph

        # Create a new DiGraph without "text_sequence" edges
        filtered_graph = nx.DiGraph()
        for u, v, data in graph.edges(data=True):
            if data.get("relation") != "text_sequence":
                # Check if the edge already exists in the filtered graph
                if filtered_graph.has_edge(u, v):
                    # If it does, increment the count of the edge
                    count = filtered_graph[u][v].get("count", 0)
                    filtered_graph[u][v]["count"] = count + 1
                else:
                    # Otherwise, add the edge to the filtered graph
                    filtered_graph.add_edge(u, v, **data)

        # Use the draw_graph function to draw the filtered graph
        self.draw_graph(graph=filtered_graph, root=root, path=path)

    def draw_graph(self, graph=None, root=None, path="graph.png"):
        if graph == None:
            graph = self.graph
        plt.figure(figsize=(10, 10), dpi=100)  # Adjust as needed
        plt.clf()  # Clear the current figure

        # pos = nx.spring_layout(graph)
        # Alternatively, try a different layout:
        pos = graphviz_layout(graph, root=root, prog="twopi")
        # pos = nx.circular_layout(self.graph)

        # color-code the edges
        color_code = {
            "the": "green",
            "ant": "red",
            "syn": "blue",
            "sub": "orange",
            "text_sequence": "gray",
        }
        edge_color_list = [color_code[rel[2]] for rel in graph.edges.data("relation")]

        nx.draw(graph, pos, with_labels=True, edge_color=edge_color_list)
        edge_labels = nx.get_edge_attributes(graph, "relation")

        leg = plt.legend(color_code, labelcolor=color_code.values())
        for i, item in enumerate(leg.legendHandles):
            item.set_color(list(color_code.values())[i])

        plt.savefig(image_folder + path, format="png", bbox_inches="tight")
        plt.close()

    def pull(self, n):
        return [
            (n, self.graph.nodes[n]["text"])
            for n in self.nodes_with_least_info(self.graph, n)
        ]

    def pull_lz(self, n, m):
        return view_shape(list(zip(*self.pull(n * m))), (2, n, m))

    yielded = []

    def nodes_with_least_info(self, G, n):
        # Dictionary to store the sum of "_score" attributes for each node
        node_scores = {}
        all_nodes = list(G.nodes())
        random.shuffle(all_nodes)
        for node in all_nodes:
            total_score = 0
            for _, _, data in G.edges(node, data=True):
                for key, value in data.items():
                    if key.endswith("_score"):
                        total_score += value
            node_scores[node] = total_score

        # Sort nodes based on their total_score
        sorted_nodes = sorted(node_scores, key=node_scores.get)
        x_sorted_nodes = sorted_nodes
        sorted_nodes = [nn for nn in sorted_nodes if nn not in self.yielded]
        if len(sorted_nodes) < n:
            self.yielded = []
            return x_sorted_nodes[:n]
        else:
            self.yielded.extend(sorted_nodes[:n])

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
            if "sub" in attr
        ]

    @staticmethod
    def computed_i_tri_out_edges(G, v, visited):
        tou = Tree.computed_tri_out_edges(G, v)
        if not tou:
            return []
        i_s = set([attr["trident"] for n1, n2, attr in tou])
        result = []
        for i in i_s:
            result.append(
                [
                    (n1, n2, attr, i)
                    for n1, n2, attr in tou
                    if attr["trident"] == i and n2 not in visited
                ]
            )
        return result

    @staticmethod
    def max_score_path(G, start=None):
        best_score = 0
        best_path_edges = []

        def edge_score(e):
            total_score = 0
            for k, v in e.items():
                if k.endswith("_score"):
                    total_score += v
            return total_score

        def dfs(v, visited, current_score, current_path_edges):
            nonlocal best_score, best_path_edges

            # if current score is worse than best, prune
            if current_score < best_score:
                return

            # Ensure not to cycle
            if v in visited:
                return

            visited.add(v)
            computed_sub_edges = Tree.computed_sub_out_edges(G, v)
            computed_tri_edges_groups = Tree.computed_i_tri_out_edges(G, v, visited)

            # If vertex fulfills the delta constraint
            for gs in computed_tri_edges_groups:
                (n1, n2, e, _i1), (n3, n4, f, _i1) = gs
                score_increment = sum(edge_score(_e) for _e in [e, f])

                visited.add(n2)
                visited.add(n4)

                dfs(
                    n2,
                    visited,
                    current_score + score_increment,
                    current_path_edges + [(n1, n2, e), (n3, n4, f)],
                )
                dfs(
                    n4,
                    visited,
                    current_score + score_increment,
                    current_path_edges + [(n1, n2, e), (n3, n4, f)],
                )

                visited.remove(n2)
                visited.remove(n4)

                # Update the best score/path if necessary
                if current_score + score_increment > best_score:
                    best_score = current_score + score_increment
                    best_path_edges = current_path_edges + [(n1, n2, e), (n3, n4, f)]

            # For regular vertices, take up to three best paths
            neighbors_sorted_by_score = sorted(
                computed_sub_edges, key=lambda e: edge_score(e), reverse=True
            )
            for sub_e in neighbors_sorted_by_score[:3]:
                (v, next_e, _attr) = sub_e
                score_increment = edge_score(_attr)
                dfs(
                    next_e,
                    visited,
                    current_score + score_increment,
                    current_path_edges + [sub_e],
                )

                # Update the best score/path if necessary
                if current_score + score_increment > best_score:
                    best_score = current_score + score_increment
                    best_path_edges = current_path_edges + [sub_e]

            visited.remove(v)

        dfs(start, set(), 0, [])

        # Convert best_path_edges to a graph
        subgraph = nx.DiGraph()
        for n1, n2, attr in best_path_edges:
            subgraph.add_edge(n1, n2, **attr)

        return subgraph

    def save_state(self, number, filename=None):
        os.makedirs(pickle_folder, exist_ok=True)
        if filename is None:
            filename = os.path.join(pickle_folder, f"tree_state_{number}.pkl")
        state = {
            "inputs": self.inputs,
            "graph": self.graph,
            "yielded": self.yielded,
        }
        with open(filename, "wb") as file:
            pickle.dump(state, file)

    @classmethod
    def load_state(cls, number=None):
        os.makedirs(pickle_folder, exist_ok=True)
        files = [f for f in os.listdir(pickle_folder) if f.startswith("tree_state_") and f.endswith(".pkl")]
        if not files:
            return None, None

        if number is None:
            numbers = [int(f.split("_")[2].split(".")[0]) for f in files]
            latest_number = max(numbers)
            filename = os.path.join(pickle_folder, f"tree_state_{latest_number}.pkl")
        else:
            filename = os.path.join(pickle_folder, f"tree_state_{number}.pkl")
            latest_number = number

        with open(filename, "rb") as file:
            state = pickle.load(file)
            tree = cls([])
            tree.inputs = state["inputs"]
            tree.graph = state["graph"]
            tree.yielded = state["yielded"]
            return tree, latest_number


if __name__ == "__main__":
    enumerated_texts = [
        ("2", "* \tDie Welt ist alles, was der Fall ist."),
        ("6", "* \tDie Welt ist alles, was der Fall ist."),
        ("5", "* \tDie Welt ist alles, was der Fall ist."),
        ("1.1", "Die Welt ist die Gesamtheit der Tatsachen, nicht der Dinge."),
    ]
    graph_wrapper = Tree(enumerated_texts)
    graph_wrapper.draw_graph()
