import hashlib
import numpy as np


class MinHash:
    def __init__(self, num_perm):
        self.num_perm = num_perm
        self.hash_funcs = self._generate_hash_funcs(num_perm)

    def _generate_hash_funcs(self, num_perm):
        hash_funcs = []
        for i in range(num_perm):
            hash_funcs.append(lambda x, i=i: int(hashlib.md5(f'{x}{i}'.encode()).hexdigest(), 16) % (2**32))
        return hash_funcs

    def compute(self, fingerprint):
        minhash = [min([func(fingerprint) for func in self.hash_funcs]) for _ in range(self.num_perm)]
        return minhash

if __name__ == "__main__":
    fingerprints = [123456, 789012, 345678]  # Example fingerprints
    minhash = MinHash(num_perm=128)
    minhashes = [minhash.compute(fp) for fp in fingerprints]
    print(minhashes)
