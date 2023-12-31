import logging

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.error("matplotlib not installed")
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


def draw_graph(G, other_attributes=None):
    # set size of figure
    plt.figure(figsize=(10, 10))
    pos = graphviz_layout(G, prog="twopi")
    nx.draw(G, pos, with_labels=True)

    # Prepare edge labels
    edge_labels = nx.get_edge_attributes(G, "relation")

    if other_attributes:
        for other_attribute in other_attributes:
            for (u, v), label in edge_labels.items():
                if other_attribute in G[u][v]:
                    additional_label = G[u][v][other_attribute]
                    edge_labels[(u, v)] = f"{label}\n{additional_label:.2f}"

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()
