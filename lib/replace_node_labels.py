import networkx as nx


def replace_colon_in_node_names(G, replacement_string):
    # Create a mapping from the old names (with colons) to the new names (with the replacement)
    mapping = {node: node.replace(".", replacement_string) for node in G.nodes()}
    # Relabel the nodes in the graph with the new names
    G_renamed = nx.relabel_nodes(G, mapping)
    return G_renamed
