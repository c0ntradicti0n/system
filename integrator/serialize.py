import logging


def serialize_graph_to_structure(graph, start_node, no_title=False):
    def get_related_nodes(node):
        """Return related nodes based on edges."""
        # Filtering edges with the current node
        edges = [
            e
            for e in graph.edges(node, data=True)
            if e[2]["relation"] in ["ant", "syn", "sub"]
        ]
        return sorted(edges, key=lambda x: x[2]["relation"])

    def get_sub_related_nodes(graph, node):
        """Return nodes related to the given node by 'sub' relation."""
        # Filtering edges with the current node that have 'sub' relation
        sub_edges = [
            e for e in graph.edges(node, data=True) if e[2]["relation"] == "sub"
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

    def construct_structure(node):
        """Recursively construct the nested structure."""
        structure = {}
        if not node:
            return structure

        structure[1] = {
            ".": node_key_text(node) + graph.nodes[node]["text"],
            **construct_structure(get_sub_related_nodes(graph, node)),
        }
        edges = get_related_nodes(node)

        for _, target, data in edges:
            if data["relation"] == "ant":
                structure[2] = {
                    ".": node_key_text(target) + graph.nodes[target]["text"],
                    **construct_structure(get_sub_related_nodes(graph, target)),
                }
            elif data["relation"] == "syn":
                structure[3] = {
                    ".": node_key_text(target) + graph.nodes[target]["text"],
                    **construct_structure(get_sub_related_nodes(graph, target)),
                }

        return structure

    return construct_structure(start_node)
