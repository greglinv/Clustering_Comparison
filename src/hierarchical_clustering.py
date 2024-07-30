import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from src.similarity import jaccard_similarity


def jaccard_distance(x, y):
    """Compute the Jaccard distance between two MinHash signatures."""
    return 1 - jaccard_similarity(x, y)


def hierarchical_clustering(minhashes):
    # Ensure MinHash signatures are 2D arrays
    minhashes_array = np.array(minhashes)

    # Calculate the pairwise Jaccard distances
    num_samples = minhashes_array.shape[0]
    jaccard_distances = np.zeros((num_samples, num_samples))


    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            jaccard_distances[i, j] = jaccard_distance(minhashes_array[i], minhashes_array[j])
            jaccard_distances[j, i] = jaccard_distances[i, j]

    # Debug: Print the Jaccard distance matrix
    print("Jaccard Distance Matrix:")
    print(jaccard_distances)

    # Convert to condensed distance matrix
    condensed_distances = squareform(jaccard_distances)

    # Debug: Print the condensed distance matrix
    print("Condensed Distance Matrix:")
    print(condensed_distances)

    # Perform hierarchical clustering
    linked = linkage(condensed_distances, method='ward')
    dendrogram(linked)
    plt.show()
    return linked


if __name__ == "__main__":
    fingerprints = [[1, 2, 3], [2, 3, 4], [4, 5, 6]]  # Example fingerprints
    minhashes = [[hash(f) for f in fp] for fp in fingerprints]
    hierarchical_clustering(minhashes)
