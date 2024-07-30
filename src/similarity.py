def jaccard_similarity(minhash1, minhash2):
    intersection = len(set(minhash1).intersection(set(minhash2)))
    union = len(set(minhash1).union(set(minhash2)))
    return intersection / union


