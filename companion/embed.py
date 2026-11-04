"""Optional local embeddings (fastembed + ONNX, no API key). Each passage is embedded on its own, so a
vector never carries information from later text; the horizon is applied when vectors are loaded."""

import os
from pathlib import Path
from typing import Protocol

DEFAULT_MODEL = os.environ.get("COMPANION_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CACHE_DIR = Path(os.environ.get("COMPANION_MODEL_DIR", Path.home() / ".cache" / "spoiler-free-companion" / "models"))


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]):
        """Return an (n, d) float32 array of L2-normalised vectors."""
        ...


class FastEmbedder:
    def __init__(self, name: str = DEFAULT_MODEL):
        self.name = name
        self._model = None

    def embed(self, texts: list[str]):
        import numpy as np

        if self._model is None:  # loaded on first use: startup stays fast when no semantic search is needed
            from fastembed import TextEmbedding

            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(self.name, cache_dir=str(CACHE_DIR))
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def default_embedder() -> Embedder | None:
    """A FastEmbedder when fastembed and numpy are installed, else None (keyword search only)."""
    try:
        import fastembed  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return None
    return FastEmbedder()
