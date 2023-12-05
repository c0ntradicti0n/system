import logging


def serialize_graph_to_structure(graph, start_node, no_title=False, depth=10):
    def get_related_nodes(node):
        """Return related nodes based on edges."""
        # Filtering edges with the current node
        edges = [
            e
            for e in graph.edges(node, data=True)
            if e[2]["relation"] in ["ant", "syn_1", "syn_2", "hie"]
        ]
        return sorted(edges, key=lambda x: x[2]["relation"])

    def get_sub_related_nodes(graph, node):
        """Return nodes related to the given node by 'sub' relation."""
        # Filtering edges with the current node that have 'sub' relation
        sub_edges = [
            e for e in graph.edges(node, data=True) if e[2]["relation"] == "hie"
        ]
        # Extracting target nodes from the edges
        sub_nodes = [target for _, target, _ in sub_edges]
        if not len(sub_nodes) in [
            1,
            0,
        ]:
            logging.error(f"sub nodes can be max only 1 in our DiGraph for {sub_nodes}")
        return sub_nodes[0] if sub_nodes else None

    def node_key_text(n):
        if no_title:
            return ""
        return f"[{n}] "

    def construct_structure(node, depth=10):
        """Recursively construct the nested structure."""
        structure = {}
        if depth == 0:
            logging.error(f"depth exhausted for {node}")
            return {}
        if not node:
            return structure

        if not node in graph.nodes:
            logging.error(f"node {node} not in graph")
            return structure

        structure[1] = {
            ".": node_key_text(node) + graph.nodes[node]["text"],
            **construct_structure(get_sub_related_nodes(graph, node)),
        }
        edges = get_related_nodes(node)

        for _, target, data in edges:
            if data["relation"] == "syn_1":
                structure[2] = {
                    ".": node_key_text(target) + graph.nodes[target]["text"],
                    **construct_structure(
                        get_sub_related_nodes(graph, target), depth - 1
                    ),
                }
            elif data["relation"] == "syn_2":
                structure[3] = {
                    ".": node_key_text(target) + graph.nodes[target]["text"],
                    **construct_structure(
                        get_sub_related_nodes(graph, target), depth - 1
                    ),
                }

        return structure

    return construct_structure(start_node)
