import networkx as nx
import matplotlib.pyplot as plt

class Tree:
    def __init__(self, enumerated_texts):
        self.graph = nx.DiGraph()
        self._populate_graph(enumerated_texts)
        self._create_sequence_edges()

    def _populate_graph(self, enumerated_texts):
        for key, text in enumerated_texts:
            self.graph.add_node(key, text=text)

    def _create_sequence_edges(self):
        sorted_nodes = sorted(self.graph.nodes())
        for i in range(len(sorted_nodes) - 1):
            self.graph.add_edge(sorted_nodes[i], sorted_nodes[i + 1], relation="text_sequence")

    def add_relation(self, key1, key2, relation_type, **kwargs):
        if key1 in self.graph and key2 in self.graph:
            self.graph.add_edge(key1, key2, relation=relation_type, **kwargs)
        else:
            print(f"Nodes {key1} or {key2} not found in the graph.")

    def filter_edges(self, relation_type):
        for u, v, data in list(self.graph.edges(data=True)):
            if data.get('relation') == relation_type:
                yield (u, v)

    def remove_edge(self, key1, key2):
        if self.graph.has_edge(key1, key2):
            self.graph.remove_edge(key1, key2)

    def draw_graph(self, graph=None, path="graph.png"):
        if not graph:
            graph =  self.graph
        plt.figure(figsize=(10, 10), dpi=100)  # Adjust as needed
        plt.clf()  # Clear the current figure

        pos = nx.spring_layout(graph)
        # Alternatively, try a different layout:
        # pos = nx.shell_layout(self.graph)
        # pos = nx.circular_layout(self.graph)

        nx.draw(graph, pos, with_labels=True)
        edge_labels = nx.get_edge_attributes(graph, 'relation')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

        plt.savefig(path, format='png', bbox_inches='tight')

    def pull(self, n):
        return [(n, self.graph.nodes[n]["text"]) for n in self.nodes_with_least_info(self.graph, n)]

    def pull_lz(self, n):
        return list(zip(*self.pull(n)))

    yielded = []
    def nodes_with_least_info(self, G, n):
        # Dictionary to store the sum of "_score" attributes for each node
        node_scores = {}
        all_nodes = list(G.nodes())
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
        if len(sorted_nodes)< n:
            self.yielded = []
            return x_sorted_nodes[:n]
        else:
            self.yielded.extend(sorted_nodes[:n])

        # Return the top n nodes with the least information
        return sorted_nodes[:n]

    @staticmethod
    def max_score_path(G, start =None):
        best_score = 0
        best_path = []
        if not start:
            start = sorted(G.degree, key=lambda x: x[1], reverse=True)
        def dfs(v, visited, current_score, current_path):
            nonlocal best_score, best_path

            # if current score is worse than best, prune
            if current_score < best_score:
                return

            # Ensure not to cycle
            if v in visited:
                return

            visited.add(v)
            neighbors = list(G.neighbors(v))

            # If vertex has the three-edge constraint
            if 'trident' in G.nodes[v]:  # Let's assume 'trident' attribute is set for such vertices
                together_edges = [e for e in neighbors if 'together' in G[v][e]]
                if len(together_edges) == 3:
                    score_increment = sum(G[v][e]['_score'] for e in together_edges)
                    dfs(together_edges[-1], visited, current_score + score_increment, current_path + together_edges)
                visited.remove(v)
                return

            # For regular vertices, take up to three best paths
            neighbors_sorted_by_score = sorted(neighbors, key=lambda e: G[v][e]['_score'], reverse=True)
            for next_v in neighbors_sorted_by_score[:3]:
                score_increment = sum(attr.get('_score', 0) for key, attr in G[v][next_v].items())
                dfs(next_v, visited, current_score + score_increment, current_path + [next_v])

            # Update the best score/path if necessary
            if current_score > best_score:
                best_score = current_score
                best_path = current_path

            visited.remove(v)

        dfs(start, set(), 0, [start])
        return best_path, best_score


if __name__ == "__main__":
    # Example usage:
    enumerated_texts = [('6', '* \tDie Welt ist alles, was der Fall ist.'),('4', '* \tDie Welt ist alles, was der Fall ist.'),('1', '* \tDie Welt ist alles, was der Fall ist.'), ('1.1', 'Die Welt ist die Gesamtheit der Tatsachen, nicht der Dinge.')]  # truncated for brevity
    graph_wrapper = Tree(enumerated_texts)
    graph_wrapper.draw_graph()