"""Semantic search finds paraphrases but can never load a passage past the horizon."""
import unittest

from companion.ingest import Chapter
from companion.store import Library
from tests.fake_embed import FakeEmbedder

BOOK = [Chapter(f"c{i}", [text]) for i, text in enumerate([
    "The little girl cried bitterly beside the grey harbour wall at dusk.",
    "A merchant counted silver coins at the market and never looked up.",
    "Much later a child wept again, tears running down beside the old river.",
], start=1)]


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.library = Library(":memory:", embedder=FakeEmbedder())
        self.library.add_work("B", BOOK)
        self.library.embed_work("B")
        self.addCleanup(self.library.close)

    def test_paraphrase_without_shared_words_is_found(self):
        found = self.library.search("B", "youngster sobbing", 10**6, limit=2)
        self.assertEqual({p.ordinal for p in found}, {1, 3})
        self.library.embedder = None  # keyword-only cannot connect the words
        self.assertEqual(self.library.search("B", "youngster sobbing", 10**6, limit=2), [])

    def test_best_semantic_match_after_the_horizon_is_never_returned(self):
        horizon = self.library.horizon_after_chapter("B", 2)
        found = self.library.search("B", "child wept tears river", horizon, limit=10)
        self.assertTrue(found)
        self.assertTrue(all(p.ordinal <= horizon for p in found))
        rows = self.library._semantic(self.library._work("B"), "child wept tears river", horizon, 10)
        self.assertTrue(all(row[0] <= horizon for row in rows))
        self.assertEqual(self.library._semantic(self.library._work("B"), "x", 0, 10), [])

    def test_exact_word_lookup_stays_first_over_semantic_neighbours(self):
        found = self.library.search("B", "merchant silver coins market", 10**6, limit=1)
        self.assertEqual([p.ordinal for p in found], [2])

    def test_embedding_is_idempotent_and_reingest_clears_vectors(self):
        self.assertEqual(self.library.embed_work("B"), 0)
        self.assertTrue(self.library.has_vectors("B"))
        self.library.add_work("B", BOOK[:1])
        self.assertFalse(self.library.has_vectors("B"))
        self.assertEqual(self.library.embed_work("B"), 1)

    def test_without_an_embedder_or_vectors_search_is_keyword_only(self):
        library = Library(":memory:")
        self.addCleanup(library.close)
        library.add_work("B", BOOK)
        self.assertFalse(library.has_vectors("B"))
        with self.assertRaises(RuntimeError):
            library.embed_work("B")
        self.assertTrue(library.search("B", "merchant", 10**6))
        library.embedder = FakeEmbedder()  # embedder set but this work has no vectors yet
        self.assertTrue(library.search("B", "merchant", 10**6))


if __name__ == "__main__":
    unittest.main()
