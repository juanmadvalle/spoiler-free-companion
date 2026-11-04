"""Anti-spoiler guarantees and ingestion, on a synthetic book (no real works)."""
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from zipfile import ZipFile

from companion import answer
from companion.cli import main
from companion.ingest import Chapter, read_book
from companion.store import Library, Passage

# Each chapter plants a unique secret word that only appears in that chapter.
SECRETS = ["alpha", "bravo", "charlie", "delta"]
CHAPTERS = [
    Chapter(f"Chapter {i}", [
        f"In chapter {i} the lighthouse keeper Marlowe found the {secret}ite crystal near the harbour. " * 2,
        f"Marlowe told Ines about the {secret}ite crystal and the strange lantern at the old harbour. " * 2,
    ])
    for i, secret in enumerate(SECRETS, start=1)
]


def make_epub(path: Path, chapters: list[tuple[str, str]]) -> None:
    items = "".join(f'<item id="c{i}" href="c{i}.xhtml" media-type="application/xhtml+xml"/>' for i in range(len(chapters)))
    refs = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
    with ZipFile(path, "w") as z:
        z.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="OEBPS/book.opf"/></rootfiles></container>')
        z.writestr("OEBPS/book.opf", f'<package xmlns="http://www.idpf.org/2007/opf"><manifest>{items}</manifest><spine>{refs}</spine></package>')
        for i, (title, body) in enumerate(chapters):
            z.writestr(f"OEBPS/c{i}.xhtml", f"<html><body><h1>{title}</h1><p>{body}</p></body></html>")


class SpoilerGuardTests(unittest.TestCase):
    def setUp(self):
        self.library = Library(":memory:")
        self.library.add_work("Synthetic", CHAPTERS)
        self.addCleanup(self.library.close)

    def test_search_never_returns_text_after_the_horizon(self):
        # Even queries naming a future secret must not surface it.
        queries = [f"what is the {s}ite crystal" for s in SECRETS] + ["Marlowe harbour lantern", "Ines"]
        for chapter in range(1, len(CHAPTERS) + 1):
            horizon = self.library.horizon_after_chapter("Synthetic", chapter)
            for query in queries:
                for passage in self.library.search("Synthetic", query, horizon, limit=50):
                    self.assertLessEqual(passage.ordinal, horizon)
                    self.assertLessEqual(passage.chapter, chapter)
                    for future in SECRETS[chapter:]:
                        self.assertNotIn(future, passage.text)

    def test_recent_never_returns_text_after_the_horizon(self):
        horizon = self.library.horizon_after_chapter("Synthetic", 2)
        text = " ".join(p.text for p in self.library.recent("Synthetic", horizon, limit=100))
        self.assertIn("bravo", text)
        self.assertNotIn("charlie", text)
        self.assertNotIn("delta", text)

    def test_model_context_contains_no_future_text(self):
        seen = []

        def spy(question, passages):
            seen.append(answer.build_context(passages))
            return "ok"

        horizon = self.library.horizon_after_chapter("Synthetic", 1)
        answer.ask(self.library, "Synthetic", "tell me about the charlieite and deltaite crystal", horizon, spy, limit=50)
        answer.recap(self.library, "Synthetic", horizon, spy, limit=50)
        self.assertEqual(len(seen), 2)
        for context in seen:
            self.assertNotIn("charlie", context)
            self.assertNotIn("delta", context)

    def test_check_horizon_fails_loudly(self):
        with self.assertRaises(AssertionError):
            answer.check_horizon([Passage(5, 2, "x", "late")], horizon=3)

    def test_hostile_query_text_cannot_break_the_filter(self):
        horizon = self.library.horizon_after_chapter("Synthetic", 1)
        for query in ['" OR 1=1 --', "NEAR(a b)", "ordinal:99 delta*", "* OR deltaite", "'; DROP TABLE chunks;--"]:
            for passage in self.library.search("Synthetic", query, horizon):
                self.assertLessEqual(passage.ordinal, horizon)

    def test_locate_finds_position_from_a_remembered_phrase(self):
        horizon, found = self.library.locate("Synthetic", "Marlowe found the bravoite crystal near the harbour")
        self.assertEqual(found.chapter, 2)
        self.assertEqual(horizon, found.ordinal)
        with self.assertRaises(LookupError):
            self.library.locate("Synthetic", "completely unrelated words zebra pineapple")

    def test_reingest_replaces_the_work(self):
        self.library.add_work("Synthetic", CHAPTERS[:1])
        self.assertEqual([c[0] for c in self.library.chapters("Synthetic")], [1])
        self.assertEqual(self.library.search("Synthetic", "charlieite", 10**6), [])


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def test_epub_follows_spine_order_and_titles(self):
        body = "Some long sentence about nothing in particular. " * 8
        make_epub(self.dir / "b.epub", [("One", body + "first"), ("Two", body + "second")])
        chapters = read_book(self.dir / "b.epub")
        self.assertEqual([c.title for c in chapters], ["One", "Two"])
        self.assertIn("first", chapters[0].paragraphs[-1])

    def test_gutenberg_text_strips_boilerplate_and_splits_chapters(self):
        filler = "A sentence that pads the paragraph out to a reasonable length for testing. " * 4
        raw = (f"Legal intro\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\nCHAPTER I\n\n{filler}\n\n"
               f"CHAPTER II\n\n{filler}secondmark\n\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nLicense text\n")
        (self.dir / "b.txt").write_text(raw, encoding="utf-8")
        chapters = read_book(self.dir / "b.txt")
        self.assertEqual([c.title for c in chapters], ["CHAPTER I", "CHAPTER II"])
        self.assertNotIn("License", " ".join(chapters[1].paragraphs))

    def test_table_of_contents_is_not_story_text(self):
        filler = "A sentence that pads the paragraph out to a reasonable length for testing. " * 4
        raw = (f"Contents I. First II. Hidden Future Title {filler}\n\nCHAPTER I\n\n{filler}\n\n"
               f"CHAPTER II\n\n{filler}\n")
        (self.dir / "b.txt").write_text(raw, encoding="utf-8")
        chapters = read_book(self.dir / "b.txt")
        self.assertEqual([c.title for c in chapters], ["CHAPTER I", "CHAPTER II"])

    def test_unsupported_format(self):
        (self.dir / "b.pdf").write_bytes(b"%PDF")
        with self.assertRaises(ValueError):
            read_book(self.dir / "b.pdf")


class CliTests(unittest.TestCase):
    def test_end_to_end_without_leaking(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            body = "Marlowe walked along the harbour and saw the {0} crystal glowing softly. " * 4
            make_epub(temp / "b.epub", [("One", body.format("firstmark")), ("Two", body.format("secondmark"))])
            db = str(temp / "lib.db")
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(main(["--db", db, "ingest", str(temp / "b.epub"), "--title", "Book"]), 0)
                out.truncate(0), out.seek(0)
                self.assertEqual(main(["--db", db, "ask", "Book", "--chapter", "1", "crystal glowing"]), 0)
                self.assertIn("firstmark", out.getvalue())
                self.assertNotIn("secondmark", out.getvalue())
                self.assertEqual(main(["--db", db, "ask", "Nope", "--chapter", "1", "x"]), 1)


if __name__ == "__main__":
    unittest.main()
