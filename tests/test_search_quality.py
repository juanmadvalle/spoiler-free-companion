"""Stemming, coverage ranking and in-place index upgrade."""
import sqlite3
import tempfile
import unittest
from pathlib import Path

from companion.ingest import Chapter
from companion.store import Library

BOOK = [Chapter("c", [p]) for p in [
    "The little girl cried bitterly beside the river while the lanterns burned low.",
    "A merchant counted silver coins at the market and never looked up from his table.",
    "The merchant and the little girl crossed the river at dawn carrying silver lanterns.",
]]  # one chapter each, so every sentence is its own passage


class SearchQualityTests(unittest.TestCase):
    def test_inflected_query_words_still_match(self):
        library = Library(":memory:")
        self.addCleanup(library.close)
        library.add_work("B", BOOK)
        found = library.search("B", "why is the girl crying", 10**6, limit=1)
        self.assertEqual([p.ordinal for p in found], [1])
        self.assertTrue(library.search("B", "burning lantern", 10**6, limit=3))

    def test_passages_covering_more_terms_rank_first(self):
        library = Library(":memory:")
        self.addCleanup(library.close)
        library.add_work("B", BOOK)
        top = library.search("B", "merchant silver lanterns river", 10**6, limit=1)
        self.assertEqual([p.ordinal for p in top], [3])

    def test_old_index_is_rebuilt_in_place(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "old.db"
            library = Library(path)
            library.add_work("B", BOOK)
            library.close()
            db = sqlite3.connect(path)  # simulate a library created before stemming
            db.executescript("DROP TABLE chunks_fts; CREATE VIRTUAL TABLE chunks_fts USING "
                             "fts5(text, content='chunks', content_rowid='id');"
                             "INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild');")
            db.close()
            library = Library(path)
            try:
                self.assertEqual([p.ordinal for p in library.search("B", "crying girl", 10**6, limit=1)], [1])
            finally:
                library.close()


if __name__ == "__main__":
    unittest.main()
