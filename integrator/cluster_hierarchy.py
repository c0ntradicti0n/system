import numpy as np

def single_linkage_clustering(data, n_clusters):
    # Calculate the Euclidean distance matrix
    distance_matrix = np.sqrt(((data[:, np.newaxis] - data[np.newaxis, :]) ** 2).sum(axis=2))

    # Initialize clusters as each individual point
    clusters = [{i} for i in range(len(data))]

    while len(clusters) > n_clusters:
        # Find the pair of clusters with minimum distance
        min_dist = np.inf
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = min(distance_matrix[x, y] for x in clusters[i] for y in clusters[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (i, j)

        # Merge the closest pair of clusters
        clusters[closest_pair[0]].update(clusters[closest_pair[1]])
        del clusters[closest_pair[1]]

    return clusters

# Example usage
data = np.array([[1, 2], [2, 3], [3, 4], [10, 10], [11, 11], [12, 12]])
n_clusters = 2
clusters = single_linkage_clustering(data, n_clusters)
print("Clusters:", clusters)
