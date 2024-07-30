import os
import numpy as np
from src.fingerprint_generator import load_fsl_fileset, generate_fingerprints
from src.minhash import MinHash
from src.similarity import jaccard_similarity
from src.hierarchical_clustering import hierarchical_clustering
from src.vae_clustering import VAE
from src.dec import DEC
from src.error_analysis import compare_methods



def main():
    # Load FSL fileset and generate fingerprints
    fsl_directory = "data/fsl_fileset"

    # Debug: Print the folder path
    print(f"Checking folder: {os.path.abspath(fsl_directory)}")

    fileset = load_fsl_fileset(fsl_directory)

    # Debug: Print loaded fileset summary
    print(f"Loaded fileset: {len(fileset)} files")

    fingerprints = generate_fingerprints(fileset)

    # Debug: Print generated fingerprints summary
    print(f"Generated fingerprints: {len(fingerprints)}")

    # Apply MinHash
    minhash = MinHash(num_perm=128)
    minhashes = [minhash.compute(fp) for fp in fingerprints]

    # Ensure MinHash signatures are 2D arrays
    minhashes = np.array(minhashes)

    # Debug: Print the MinHash signatures summary
    print(f"MinHash Signatures: {minhashes.shape}")

    # Similarity Measurement
    similarity_matrix = [
        [jaccard_similarity(minhashes[i], minhashes[j]) for j in range(len(minhashes))]
        for i in range(len(minhashes))
    ]

    # Debug: Print the Jaccard Distance Matrix summary
    print("Jaccard Distance Matrix (truncated):")
    print(np.array(similarity_matrix)[:5, :5])  # Print only a part of the matrix for brevity

    # Clustering and Comparison
    results = compare_methods(minhashes)  # Pass minhashes to compare_methods

    print("\nComparison Results Summary:")
    print(f"Hierarchical Clustering Time: {results['Hierarchical Clustering']['time']:.4f} seconds")
    print("Hierarchical Clustering Dendrogram:")
    print(results['Hierarchical Clustering']['clusters'])

    print(f"\nVAE Clustering Time: {results['VAE Clustering']['time']:.4f} seconds")
    print("VAE Clustering Results:")
    print(results['VAE Clustering']['clusters'])


if __name__ == "__main__":
    main()
