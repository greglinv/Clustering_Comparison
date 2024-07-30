import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def hierarchical_clustering(fingerprints):
    """
    Perform hierarchical clustering on fingerprints.

    :param fingerprints: List of fingerprints
    :return: Cluster labels
    """
    condensed_distances = pdist(fingerprints, metric='jaccard')
    linked = linkage(condensed_distances, method='ward')

    # Determine the number of clusters based on the maximum distance
    max_d = 0.5  # This is a threshold value; you might need to adjust it
    clusters = fcluster(linked, max_d, criterion='distance')

    # Ensure we return the correct number of clusters
    num_clusters = len(set(clusters))
    if num_clusters != len(fingerprints):
        print(f"Expected number of clusters: {len(fingerprints)}, but got: {num_clusters}. Adjusting...")
        clusters = fcluster(linked, t=len(fingerprints), criterion='maxclust')

    return clusters - 1  # Adjust cluster labels to be 0-based
