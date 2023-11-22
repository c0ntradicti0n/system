import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


def draw_graph(G, other_attributes=None):
    # set size of figure
    plt.figure(figsize=(10, 10))
    pos = graphviz_layout(G, prog="twopi")
    nx.draw(G, pos, with_labels=True)
    # Prepare edge labels
    # Prepare edge labels
    edge_labels = nx.get_edge_attributes(G, "relation")

    if other_attributes:
        for (u, v), label in edge_labels.items():
            if other_attributes in G[u][v]:
                additional_label = G[u][v][other_attributes]
                edge_labels[(u, v)] = f"{label}\n{additional_label:.2f}"

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()
