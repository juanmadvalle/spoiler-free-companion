"""Regression tests for the code-review findings: aliases, agent card JSON, CLI chapter 0, subtitle chapters."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from companion.agent import MAX_RESULT_CHARS, Session, card_json
from companion.cli import main
from companion.ingest import Chapter, read_subtitles, subtitle_chapter
from companion.names import expand_words, extract_names
from companion.store import Library

PAD = " The evening wind carried the smell of salt across the grey harbour while the lamplighter climbed his ladder."


class AliasTimingTests(unittest.TestCase):
    CHUNKS = [(1, "Yesterday we met Holmes and Sherlock Holmes said nothing at all today."),
              (2, "The lamp burned low."), (3, "Holmes sat by the window quietly in the dark room."),
              (4, "Then Sherlock Holmes spoke again about the case and Holmes nodded twice."),
              (5, "Holmes went home but Sherlock Holmes stayed behind as Holmes often did.")]

    def test_a_name_is_established_when_it_has_been_seen_enough_times_not_at_its_first_sighting(self):
        names = extract_names(self.CHUNKS)
        self.assertEqual(names, {("sherlock", "holmes"): 4})  # first seen at 1, established at its second sighting
        known = lambda horizon: {n: s for n, s in names.items() if s <= horizon}
        self.assertEqual(expand_words(["sherlock"], known(3), 3), [])  # the justifying passage is still ahead
        self.assertEqual(expand_words(["sherlock"], known(4), 4), ["holmes"])

    def test_search_does_not_expand_a_query_before_the_second_sighting(self):
        # Anna and Reed are each mentioned often in chapter 1, but the full name "Anna Reed" appears once there
        chapters = [Chapter("One", [f"Yesterday Marlow met Anna Reed at the quay. {PAD}",
                                    f"Marlow thanked Anna warmly, and later Reed arrived. {PAD}",
                                    f"Everyone waved to Anna and to Reed. {PAD}"]),
                    Chapter("Two", [f"Later Marlow bought bread from Anna Reed again. {PAD}"])]
        library = Library(":memory:")
        self.addCleanup(library.close)
        library.add_work("B", chapters)
        first = library.horizon_after_chapter("B", 1)
        last = library.horizon_after_chapter("B", 2)
        self.assertEqual(library.aliases("B", ["reed"], first), [])
        self.assertEqual(library.aliases("B", ["reed"], last), ["anna"])


class AgentCardJsonTests(unittest.TestCase):
    @staticmethod
    def huge_card():
        text = "A very long sentence about the character that keeps going and going. " * 12
        item = {"ordinal": 5, "chapter": 1, "chapter_title": "One", "text": text, "start": None}
        return {"name": "Holmes", "aliases": ["Sherlock"], "kind": "character", "mentions": 99, "passages": 50,
                "chapters": [1, 2, 3], "first": dict(item), "last": dict(item), "described_as": [dict(item)] * 3,
                "related": [{"name": f"Someone{i}", "together": 9 - i} for i in range(6)], "position": 100}

    def test_card_json_always_parses_and_fits(self):
        card = self.huge_card()
        self.assertGreater(len(json.dumps(card)), 2000)
        for limit in (MAX_RESULT_CHARS, 1500, 700, 300):
            text = card_json(card, limit)
            self.assertEqual(json.loads(text)["name"], "Holmes")  # valid JSON even when squeezed hard
            if limit >= 700:
                self.assertLessEqual(len(text), limit)

    def test_a_small_card_is_returned_unchanged(self):
        small = {**self.huge_card(), "described_as": [], "related": []}
        small["first"]["text"] = small["last"]["text"] = "Short."
        self.assertEqual(json.loads(card_json(small)), small)

    def test_the_agent_tool_returns_parseable_json(self):
        chapters = [Chapter("One", [f"Yesterday Marlow talked to Ines, the keeper of the light, and later Marlow asked Ines. {PAD}"] * 4)]
        library = Library(":memory:")
        self.addCleanup(library.close)
        library.add_work("B", chapters)
        result = Session(library, "B", library.horizon_after_chapter("B", 1)).run_tool("character_card", {"name": "Marlow"})
        self.assertEqual(json.loads(result)["name"], "Marlow")


class CliChapterZeroTests(unittest.TestCase):
    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--no-embeddings", *args])
        return code, out.getvalue(), err.getvalue()

    def test_chapter_zero_means_nothing_read_yet_like_the_web_page(self):
        with tempfile.TemporaryDirectory() as folder:
            db = str(Path(folder) / "lib.db")
            library = Library(db)
            library.add_work("B", [Chapter("One", [f"Marlow talked to Ines at the harbour. {PAD}" * 4])])
            library.close()
            code, out, err = self.run_cli("--db", db, "characters", "B", "--chapter", "0")
            self.assertEqual((code, out.strip()), (0, ""), err)
            code, out, err = self.run_cli("--db", db, "ask", "B", "Marlow", "--chapter", "0")
            self.assertEqual(code, 0, err)
            self.assertIn("Nothing relevant", out)
            code, out, err = self.run_cli("--db", db, "ask", "B", "Marlow", "--chapter", "-1")
            self.assertEqual(code, 1)  # a negative chapter is still an error


class SubtitleChapterTests(unittest.TestCase):
    SRT = ("1\n00:00:01,000 --> 00:00:03,000\nHello there friend.\n\n"
           "2\n00:00:50,000 --> 00:00:52,000\nA much later line of dialogue.\n")

    def test_file_and_text_paths_build_the_same_chapter(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "show.S01E03.srt"
            path.write_text(self.SRT, encoding="utf-8")
            from_file = read_subtitles(path)
            from_text = subtitle_chapter(self.SRT, "show.S01E03.srt")
        self.assertEqual(from_file, from_text)
        self.assertEqual(from_text.title, "S01E03")
        self.assertEqual(len(from_text.paragraphs), len(from_text.times))
        self.assertEqual(Chapter.timed("X", [("a b", 1.0, 2.0)]), Chapter("X", ["a b"], [(1.0, 2.0)]))

    def test_empty_subtitles_are_an_error_with_the_file_name(self):
        with self.assertRaisesRegex(ValueError, "empty.srt"):
            subtitle_chapter("nothing useful", "empty.srt")


if __name__ == "__main__":
    unittest.main()
