import numpy as np


def neighbor_joining(distance_matrix):
    # Convert the distance matrix to float if it's not already
    distance_matrix = np.array(distance_matrix, dtype=float)

    n = len(distance_matrix)
    tree = {}
    clusters = {i: [i] for i in range(n)}
    active_indices = list(range(n))

    while len(distance_matrix) > 2:
        # Calculate Q-matrix
        total_distances = np.sum(distance_matrix, axis=1)
        q_matrix = (
            (n - 2) * distance_matrix - total_distances[:, None] - total_distances
        )

        # Set the diagonal to a large number to avoid selecting it
        np.fill_diagonal(q_matrix, np.inf)

        # Find the pair with the smallest distance in Q-matrix
        i, j = np.unravel_index(np.argmin(q_matrix, axis=None), q_matrix.shape)

        # Map i, j to original indices
        orig_i, orig_j = active_indices[i], active_indices[j]

        # Calculate distances for new node
        new_distances = (
            distance_matrix[i] + distance_matrix[j] - distance_matrix[i, j]
        ) / 2

        # Update tree
        new_node = max(clusters.keys()) + 1
        tree[new_node] = (clusters[orig_i], clusters[orig_j], distance_matrix[i, j])

        # Update clusters
        clusters[new_node] = clusters[orig_i] + clusters[orig_j]
        # Update clusters
        clusters[new_node] = clusters.get(orig_i, []) + clusters.get(orig_j, [])
        if orig_i in clusters:
            del clusters[orig_i]
        if orig_j in clusters:
            del clusters[orig_j]

        # Update distance matrix
        distance_matrix = np.delete(distance_matrix, [i, j], axis=0)
        distance_matrix = np.delete(distance_matrix, [i, j], axis=1)

        # Correctly compute the new row and column for the distance matrix
        new_row = np.insert(
            new_distances, min(i, j), 0
        )  # Insert 0 at the position of the smaller index
        if i != j:
            new_row = np.insert(
                new_row, max(i, j) - 1, 0
            )  # Adjust for the deletion of the first index

        # Ensure the new row has the correct length
        new_row = new_row[: len(distance_matrix)]

        # Add the new row and column to the distance matrix
        distance_matrix = np.vstack([distance_matrix, new_row])
        new_column = np.append(new_row, 0)  # Append 0 for the distance to itself
        distance_matrix = np.column_stack([distance_matrix, new_column])

        # Update active indices
        active_indices.remove(orig_i)
        active_indices.remove(orig_j)
        active_indices.append(new_node)

        # Update n
        n -= 1

    # Add the last two remaining clusters
    i, j = active_indices[0], active_indices[1]
    tree[len(tree)] = (clusters[i], clusters[j], distance_matrix[0, 1])

    return tree


# Example distance matrix
distance_matrix = np.array(
    [
        [0, 5, 9, 9, 8],
        [5, 0, 10, 10, 9],
        [9, 10, 0, 8, 7],
        [9, 10, 8, 0, 3],
        [8, 9, 7, 3, 0],
    ]
)

# Run the algorithm
tree = neighbor_joining(distance_matrix)
print(tree)
