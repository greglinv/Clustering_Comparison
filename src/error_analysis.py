import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from src.hierarchical_clustering import hierarchical_clustering
from src.vae_clustering import VAE
from src.dec import DEC
from src.fingerprint_generator import load_fsl_fileset, generate_fingerprints


def evaluate_clustering(true_labels, predicted_labels, data):
    # Check if the number of true labels and predicted labels are the same
    print(f"Length of true_labels: {len(true_labels)}")
    print(f"Length of predicted_labels: {len(predicted_labels)}")

    assert len(true_labels) == len(predicted_labels), "Number of true labels and predicted labels must be the same"

    accuracy = accuracy_score(true_labels, predicted_labels)
    rand_index = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    silhouette = silhouette_score(data, predicted_labels)
    calinski_harabasz = calinski_harabasz_score(data, predicted_labels)
    davies_bouldin = davies_bouldin_score(data, predicted_labels)
    return accuracy, rand_index, nmi, silhouette, calinski_harabasz, davies_bouldin


def compare_methods(fingerprints):
    # Hierarchical Clustering
    start_time = time.time()
    hierarchical_clusters = hierarchical_clustering(fingerprints)
    hierarchical_time = time.time() - start_time

    # VAE Clustering
    start_time = time.time()
    vae = VAE(input_dim=len(fingerprints[0]), latent_dim=10)
    vae.train(np.array(fingerprints))
    dec = DEC(vae, n_clusters=10)
    vae_clusters = dec.cluster(np.array(fingerprints))
    vae_time = time.time() - start_time

    # Ensure the correct number of true labels
    true_labels = np.arange(len(fingerprints))

    # Debug: Print lengths of labels and clusters
    print(f"Length of true_labels: {len(true_labels)}")
    print(f"Length of hierarchical_clusters: {len(hierarchical_clusters)}")
    print(f"Length of vae_clusters: {len(vae_clusters)}")

    hierarchical_metrics = evaluate_clustering(true_labels, hierarchical_clusters, fingerprints)
    vae_metrics = evaluate_clustering(true_labels, vae_clusters, fingerprints)

    return {
        "Hierarchical Clustering": {
            "time": hierarchical_time,
            "clusters": hierarchical_clusters,
            "metrics": hierarchical_metrics,
        },
        "VAE Clustering": {
            "time": vae_time,
            "clusters": vae_clusters,
            "metrics": vae_metrics,
        },
    }


if __name__ == "__main__":
    fingerprints = generate_fingerprints(load_fsl_fileset("../data/fsl_fileset"))
    results = compare_methods(fingerprints)
    print(results)
