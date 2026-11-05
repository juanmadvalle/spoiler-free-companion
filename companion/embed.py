"""Optional local embeddings (fastembed + ONNX, no API key). Each passage is embedded on its own, so a
vector never carries information from later text; the horizon is applied when vectors are loaded."""

import os
from typing import Protocol

from .paths import model_cache

DEFAULT_MODEL = os.environ.get("COMPANION_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MULTILINGUAL_MODEL = os.environ.get("COMPANION_EMBED_MODEL_MULTILINGUAL",
                                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RERANK_MODEL = os.environ.get("COMPANION_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]):
        """Return an (n, d) float32 array of L2-normalised vectors."""
        ...


class Reranker(Protocol):
    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score per text (higher is better)."""
        ...


class FastEmbedder:
    def __init__(self, name: str = DEFAULT_MODEL):
        self.name = name
        self._model = None

    def embed(self, texts: list[str]):
        import numpy as np

        if self._model is None:  # loaded on first use: startup stays fast when no semantic search is needed
            from fastembed import TextEmbedding

            cache = model_cache()
            cache.mkdir(parents=True, exist_ok=True)
            self._model = TextEmbedding(self.name, cache_dir=str(cache))
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def embed_query(self, texts: list[str]):
        """Queries get the instruction prefix BGE and Arctic models were trained with; passages do not."""
        if self.name.startswith(("BAAI/bge", "snowflake/snowflake-arctic-embed")):
            texts = [BGE_QUERY_PREFIX + t for t in texts]
        return self.embed(texts)


class FastReranker:
    """Cross-encoder that reads the question and a passage together (slower, more precise)."""

    def __init__(self, name: str = RERANK_MODEL):
        self.name = name
        self._model = None

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            cache = model_cache()
            cache.mkdir(parents=True, exist_ok=True)
            self._model = TextCrossEncoder(self.name, cache_dir=str(cache))
        return [float(score) for score in self._model.rerank(query, texts)]


class LanguageEmbedders:
    """Picks the embedding model for a book's language: English-only BGE, or a multilingual model otherwise."""

    def __init__(self):
        self._models: dict[str, FastEmbedder] = {}

    def for_lang(self, language: str) -> FastEmbedder:
        name = DEFAULT_MODEL if language == "en" else MULTILINGUAL_MODEL
        return self._models.setdefault(name, FastEmbedder(name))


def default_embedder() -> LanguageEmbedders | None:
    """Language-aware embedders when fastembed and numpy are installed, else None (keyword search only)."""
    try:
        import fastembed  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return None
    return LanguageEmbedders()


def default_reranker() -> Reranker | None:
    try:
        import fastembed  # noqa: F401
    except ImportError:
        return None
    return FastReranker()
