"""Anti-spoiler checks on real public-domain books (Project Gutenberg), downloaded once and cached.

Skipped when a book is not cached and cannot be downloaded.
"""
import re
import unittest
import urllib.request
from pathlib import Path

from companion import answer
from companion.agent import Session
from companion.ingest import chunk, read_book
from companion.store import Library, _words

DATA = Path(__file__).parent / "data"
BOOKS = {  # Gutenberg id -> (title, expected chapter count)
    1661: ("Sherlock Holmes", 12),
    1342: ("Pride and Prejudice", 61),
    11: ("Alice in Wonderland", 12),
    84: ("Frankenstein", 28),  # 4 letters + 24 chapters
    320: ("Lazarillo de Tormes", 7),  # Spanish: seven "tratados"
    2000: ("Don Quijote", 126),  # Spanish: 52 + 74 chapters
}
MIN_WORD = 5
SAMPLE = 40


def fetch(book_id: int) -> Path | None:
    path = DATA / f"pg{book_id}.txt"
    if not path.exists():
        try:
            DATA.mkdir(exist_ok=True)
            with urllib.request.urlopen(f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt", timeout=30) as r:
                path.write_bytes(r.read())
        except OSError:
            return None
    return path


def strings(value):
    """Every string value (not key) inside nested lists and dicts."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if key != "kind":  # a label made by the program ("character"/"other"), not text from the book
                yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def spread(items: list, count: int) -> list:
    """Deterministic, evenly spaced sample."""
    step = max(1, len(items) // count)
    return items[::step][:count]


class RealBookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.books = {}
        for book_id, (title, _) in BOOKS.items():
            path = fetch(book_id)
            if path:
                chapters = read_book(path)
                library = Library(":memory:")
                library.add_work(title, chapters)
                words = [{w for p in c.paragraphs for w in _words(p) if len(w) >= MIN_WORD} for c in chapters]
                cls.books[title] = (library, chapters, words)

    @classmethod
    def tearDownClass(cls):
        for library, _, _ in cls.books.values():
            library.close()

    def each_book(self):
        if not self.books:
            self.skipTest("Real books unavailable (offline and not cached)")
        for title, data in self.books.items():
            with self.subTest(book=title):
                yield title, *data

    def test_chapter_counts_and_no_boilerplate(self):
        expected = {title: count for title, count in BOOKS.values()}
        for title, library, chapters, _ in self.each_book():
            self.assertEqual(len(chapters), expected[title])
            everything = " ".join(p for c in chapters for p in c.paragraphs).lower()
            for banned in ("project gutenberg", "saintsbury"):
                self.assertNotIn(banned, everything)

    def test_future_only_vocabulary_never_reaches_any_read_path(self):
        for title, library, chapters, words in self.each_book():
            for k in spread(list(range(1, len(chapters))), 15):
                seen = set().union(*words[:k])
                future_only = sorted(set().union(*words[k:]) - seen)
                self.assertTrue(future_only, f"no future-only words after chapter {k}")
                horizon = library.horizon_after_chapter(title, k)
                session = Session(library, title, horizon)
                queries = [" ".join(spread(future_only, SAMPLE))] + spread(future_only, SAMPLE)
                results = []
                for query in queries:
                    results += library.search(title, query, horizon, limit=50)
                    results += answer_context_passages(library, title, query, horizon)
                    session.run_tool("search_passages", {"query": query, "horizon": 10**9})
                results += library.recent(title, horizon, limit=500)
                for chapter in range(k + 1, len(chapters) + 1):
                    self.assertEqual(library.chapter_passages(title, chapter, horizon, limit=500), [])
                    self.assertNotIn("[", session.run_tool("read_chapter", {"chapter": chapter})[:1])
                for passage in results:
                    self.assertLessEqual(passage.ordinal, horizon)
                    self.assertFalse(set(_words(passage.text)) & set(future_only),
                                     f"future word leaked after chapter {k}")

    def test_semantic_path_never_loads_future_passages(self):
        from tests.fake_embed import FakeEmbedder

        for title, library, chapters, words in self.each_book():
            library.embedder = FakeEmbedder()
            self.addCleanup(setattr, library, "embedder", None)
            library.embed_work(title)
            loaded = []
            original = library._semantic

            def recording(*args, **kwargs):
                rows = original(*args, **kwargs)
                loaded.extend(rows)
                return rows

            library._semantic = recording
            self.addCleanup(setattr, library, "_semantic", original)
            for k in spread(list(range(1, len(chapters))), 10):
                seen = set().union(*words[:k])
                future_only = sorted(set().union(*words[k:]) - seen)
                horizon = library.horizon_after_chapter(title, k)
                for query in (" ".join(spread(future_only, SAMPLE)), *spread(future_only, 10)):
                    for passage in library.search(title, query, horizon, limit=50):
                        self.assertLessEqual(passage.ordinal, horizon)
                        self.assertFalse(set(_words(passage.text)) & set(future_only))
                self.assertTrue(loaded)
                self.assertTrue(all(row[0] <= horizon for row in loaded))
                loaded.clear()

    def test_character_cards_contain_no_future_only_words(self):
        """Every name, alias, excerpt and related name in any card must exist at or before the reader's position."""
        for title, library, chapters, words in self.each_book():
            checked = 0
            for k in spread(list(range(1, len(chapters))), 10):
                titles = {w for c in chapters[:k] for w in _words(c.title)}  # headings of chapters already reached
                seen = set().union(*words[:k]) | titles
                future_only = set().union(*words[k:]) - seen
                horizon = library.horizon_after_chapter(title, k)
                people = library.characters(title, horizon)
                self.assertTrue(people, f"no names after chapter {k}")
                outputs = [people]
                for person in people[:8]:
                    card = library.card(title, person["name"], horizon)
                    outputs.append(card)
                    for related in card["related"][:2]:
                        outputs.append(library.card(title, related["name"], horizon))
                    checked += 1
                leaked = {w for text in strings(outputs) for w in _words(text)} & future_only
                self.assertFalse(leaked, f"{title} after chapter {k}: {sorted(leaked)[:5]}")
                self.assertFalse({c["name"].lower() for c in people} & future_only)  # a later name has no entry yet
            self.assertGreater(checked, 10)

    def test_first_appearance_traps(self):
        """A character is invisible until the chapter they first appear in."""
        traps = {"Sherlock Holmes": "jabez", "Pride and Prejudice": "wickham", "Frankenstein": "clerval",
                 "Alice in Wonderland": "caterpillar",
                 "Lazarillo de Tormes": "escudero", "Don Quijote": "sancho"}
        for title, library, chapters, _ in self.each_book():
            word = traps[title]
            first = next(i for i, c in enumerate(chapters, 1) if word in " ".join(c.paragraphs).lower())
            self.assertGreater(first, 1)
            before = library.horizon_after_chapter(title, first - 1)
            after = library.horizon_after_chapter(title, first)
            self.assertEqual(library.search(title, word, before), [])
            self.assertTrue(library.search(title, word, after))

    def test_alias_expansion_only_uses_names_already_seen(self):
        for title, library, chapters, _ in self.each_book():
            names = library.db.execute(
                "SELECT n.name, n.first_seen FROM names n JOIN works w ON w.id = n.work_id WHERE w.title = ?",
                (title,)).fetchall()
            tokens = sorted({t for name, _ in names for t in name.split()})
            for k in spread(list(range(1, len(chapters) + 1)), 15):
                horizon = library.horizon_after_chapter(title, k)
                allowed = {t for name, seen in names if seen <= horizon for t in name.split()}
                for token in tokens:
                    extra = library.aliases(title, [token], horizon)
                    self.assertLessEqual(set(extra), allowed, f"{token} after chapter {k}")
                    if token not in allowed:
                        self.assertEqual(extra, [], f"{token} expanded before its name appeared")

    def test_sherlock_first_name_finds_surname_only_passages(self):
        if "Sherlock Holmes" not in self.books:
            self.skipTest("Sherlock Holmes unavailable")
        library = self.books["Sherlock Holmes"][0]
        horizon = library.horizon_after_chapter("Sherlock Holmes", 1)
        found = library.search("Sherlock Holmes", "Sherlock", horizon, limit=50)
        self.assertTrue(any("sherlock" not in p.text.lower() and "holmes" in p.text.lower() for p in found))

    def test_locating_a_remembered_phrase_finds_the_right_chapter(self):
        total = exact = later = 0
        for title, library, chapters, _ in self.each_book():
            for number, chapter in enumerate(chapters, 1):
                for text in spread(chunk(chapter.paragraphs), 3):
                    phrase = " ".join(text.split()[:14])
                    if len(phrase.split()) < 10:
                        continue
                    try:
                        _, found = library.locate(title, phrase)
                    except LookupError:
                        total += 1
                        continue
                    total += 1
                    exact += found.chapter == number
                    later += found.chapter > number
        print(f"\nlocate: {exact}/{total} exact, {later} placed too late")
        self.assertGreaterEqual(exact / total, 0.95)
        self.assertLessEqual(later / total, 0.02)


def answer_context_passages(library, title, query, horizon):
    seen = []
    answer.ask(library, title, query, horizon, lambda q, p: seen.extend(p) or "", limit=50)
    return seen


if __name__ == "__main__":
    unittest.main()
