import time
import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score
from src.hierarchical_clustering import hierarchical_clustering
from src.vae_clustering import VAE
from src.dec import DEC
from src.fingerprint_generator import load_fsl_fileset, generate_fingerprints


def evaluate_clustering(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    rand_index = adjusted_rand_score(true_labels, predicted_labels)
    return accuracy, rand_index

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

    return {
        "Hierarchical Clustering": {"time": hierarchical_time, "clusters": hierarchical_clusters},
        "VAE Clustering": {"time": vae_time, "clusters": vae_clusters}
    }

if __name__ == "__main__":
    fingerprints = generate_fingerprints(load_fsl_fileset("../data/fsl_fileset"))
    results = compare_methods(fingerprints)
    print(results)
