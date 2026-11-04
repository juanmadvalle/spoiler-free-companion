"""Deterministic stand-in for a real embedding model: words in the same group share a direction."""
import hashlib

import numpy as np

DIM = 64
GROUPS = [
    {"cried", "crying", "weeping", "wept", "tears", "sobbing"},
    {"child", "girl", "kid", "youngster"},
    {"merchant", "trader", "seller"},
    {"river", "stream", "water"},
]
_GROUP_OF = {w: i for i, g in enumerate(GROUPS) for w in g}


class FakeEmbedder:
    name = "fake-hash-embedder"

    def embed(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().replace(".", " ").replace(",", " ").split():
                if word in _GROUP_OF:
                    out[row, _GROUP_OF[word]] += 1.0
                else:
                    out[row, 8 + int(hashlib.md5(word.encode()).hexdigest(), 16) % (DIM - 8)] += 0.3
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
