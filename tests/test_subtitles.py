"""Films and series from subtitles: parsing, time positions and spoiler safety at the minute."""
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from companion import subtitles
from companion.cli import main
from companion.ingest import Chapter, read_book, read_subtitles
from companion.store import Library
from tests.test_real_books import strings
from tests.test_web import WebCase

DATA = Path(__file__).parent / "data"


def srt(cues, start_number=1):
    """Cues as (start, end, text) seconds to SRT text."""
    def stamp(t):
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(round((t % 1) * 1000)):03d}"

    return "\n\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}" for i, (a, b, text) in enumerate(cues, start_number)) + "\n"


class ParsingTests(unittest.TestCase):
    def test_srt_with_bom_crlf_markup_entities_ads_and_unsorted_cues(self):
        text = ("﻿2\r\n00:00:05,000 --> 00:00:07,500\r\n<i>Second &amp; last</i>\r\n\r\n"
                "1\r\n00:00:01,000 --> 00:00:03,000\r\n{\\an8}- Hello there.\r\n\r\n"
                "3\r\n00:00:08,000 --> 00:00:09,000\r\nSubtitles by OpenSubtitles.org\r\n\r\n"
                "4\r\n00:00:10,000 --> 00:00:11,000\r\n\r\n\r\n"
                "5\r\n00:01:02,5 --> 00:01:04,000\r\n♪ La la ♪\r\n")
        cues = subtitles.parse_cues(text)
        self.assertEqual([(c.start, c.text) for c in cues], [(1.0, "Hello there."), (5.0, "Second & last"), (62.5, "La la")])

    def test_webvtt_without_hours_and_with_notes(self):
        text = ("WEBVTT\n\nNOTE written by a tool\n\nSTYLE\n::cue { color: red }\n\n"
                "01:02.250 --> 01:04.000 align:start\nLine one\nline two\n\n"
                "1:00:01.000 --> 1:00:02.000\nAn hour in\n")
        cues = subtitles.parse_cues(text)
        self.assertEqual([(c.start, c.text) for c in cues], [(62.25, "Line one line two"), (3601.0, "An hour in")])

    def test_old_windows_encoding_and_clocks(self):
        raw = "1\n00:00:01,000 --> 00:00:02,000\nCafé señor\n".encode("cp1252")
        self.assertIn("Café señor", subtitles.decode(raw))
        self.assertEqual([subtitles.parse_clock(v) for v in ("90", "12:30", "1:02:03", "1h02m03s", "5m")], [90, 750, 3723, 3723, 300])
        for bad in ("", "abc", "1:2:3:4", "-5"):
            with self.assertRaises(ValueError):
                subtitles.parse_clock(bad)
        self.assertEqual((subtitles.format_clock(750), subtitles.format_clock(3723)), ("12:30", "1:02:03"))

    def test_passages_respect_size_duration_and_silence(self):
        cues = [subtitles.Cue(i * 2.0, i * 2.0 + 1.5, f"Line number {i} says something fairly long here.") for i in range(40)]
        cues += [subtitles.Cue(500.0, 502.0, "After a long silence."), subtitles.Cue(503.0, 504.0, "Same scene.")]
        grouped = subtitles.passages(cues)
        self.assertTrue(all(len(text) <= subtitles.MAX_CHARS for text, _, _ in grouped))
        self.assertTrue(all(end - start <= subtitles.MAX_SPAN + 2 for _, start, end in grouped))
        self.assertEqual(grouped[-1][0], "After a long silence. Same scene.")  # a silence starts a new passage
        starts = [start for _, start, _ in grouped]
        self.assertEqual(starts, sorted(starts))
        self.assertEqual(" ".join(text for text, _, _ in grouped), " ".join(c.text for c in cues))

    def test_episode_titles_and_natural_order(self):
        self.assertEqual(subtitles.episode_title("Show.S01E02.1080p.srt", 5), "S01E02")
        self.assertEqual(subtitles.episode_title("show 2x13.srt", 5), "S02E13")
        self.assertEqual(subtitles.episode_title("The_Pilot.srt", 1), "The Pilot")
        names = ["ep10.srt", "ep2.srt", "ep1.srt"]
        self.assertEqual(sorted(names, key=subtitles.natural_key), ["ep1.srt", "ep2.srt", "ep10.srt"])

    def test_read_book_accepts_a_single_subtitle_file_as_a_film(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "film.srt"
            path.write_text(srt([(1, 3, "Hello there friend."), (4, 6, "How are you today?")]), encoding="utf-8")
            chapters = read_book(path)
            self.assertEqual(len(chapters), 1)
            self.assertEqual(chapters[0].times, [(1.0, 6.0)])
            empty = Path(folder) / "empty.srt"
            empty.write_text("nothing here", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_subtitles(empty)


def timed_series():
    """Two episodes; each passage is one 10-second cue. A secret word sits in the final passage of episode 2."""
    def episode(title, lines):
        times = [(i * 20.0, i * 20.0 + 10.0) for i in range(len(lines))]  # one passage per line, 10 s long
        return Chapter(title, lines, times)

    pad = " The evening wind carried the smell of salt across the grey harbour tonight."
    one = episode("S01E01", [f"Captain Marlow arrived at the lighthouse and met Ines.{pad}",
                             f"Ines said the lantern was old and Marlow asked about it.{pad}",
                             f"Marlow found the lantern in the cellar with Ines.{pad}"])
    two = episode("S01E02", [f"Ines told Marlow about Captain Aldrich.{pad}",
                             f"Marlow and Ines argued by the sea.{pad}",
                             f"At last Zephyrine appeared, and Zephyrine was the traitor.{pad}"])
    return [one, two]


class TimedStoreTests(unittest.TestCase):
    def setUp(self):
        self.library = Library(":memory:")
        self.library.add_work("Show", timed_series())
        self.addCleanup(self.library.close)
        self.rows = self.library.db.execute("SELECT ordinal, chapter, start_s, end_s FROM chunks ORDER BY ordinal").fetchall()

    def test_passages_remember_their_times(self):
        self.assertTrue(self.library.is_timed("Show"))
        self.assertEqual(len(self.rows), 6)
        self.assertEqual(self.library.times("Show", [1, 6]), {1: (0.0, 10.0), 6: (40.0, 50.0)})
        book = Library(":memory:")
        self.addCleanup(book.close)
        book.add_work("Book", [Chapter("One", ["Plain text of a book chapter. " * 20])])
        self.assertFalse(book.is_timed("Book"))
        self.assertEqual(book.times("Book", [1]), {})
        with self.assertRaises(ValueError):
            book.horizon_at_time("Book", 1, 30)

    def test_position_counts_only_passages_that_have_fully_ended(self):
        at = lambda episode, seconds: self.library.horizon_at_time("Show", episode, seconds)
        self.assertEqual(at(1, 0), 0)  # the first line has not finished
        self.assertEqual(at(1, 9.9), 0)
        self.assertEqual(at(1, 10), 1)  # a passage that ends exactly now has been heard
        self.assertEqual(at(1, 45), 2)
        self.assertEqual(at(1, 10_000), 3)  # past the end of the episode
        self.assertEqual(at(2, 0), 3)  # start of episode 2: all of episode 1
        self.assertEqual(at(2, 31), 5)
        for episode in (0, 3, -1):
            with self.assertRaises(ValueError):
                at(episode, 0)

    def test_no_read_path_shows_anything_after_the_minute(self):
        horizon = self.library.horizon_at_time("Show", 2, 35)  # inside the second passage of episode 2
        self.assertEqual(horizon, 5)  # passages 4 and 5 ended by 0:35; passage 6 starts at 0:40
        blobs = [json.dumps([p.__dict__ for p in self.library.search("Show", q, horizon, limit=20)])
                 for q in ("Zephyrine traitor", "who is the traitor", "Ines Marlow sea")]
        blobs.append(json.dumps([p.__dict__ for p in self.library.recent("Show", horizon, 50)]))
        blobs.append(json.dumps(self.library.characters("Show", horizon)))
        blobs.append(json.dumps(self.library.card("Show", "Ines", horizon)))
        for blob in blobs:
            self.assertNotIn("Zephyrine", blob)
            self.assertNotIn("traitor", blob)
        self.assertIn("Zephyrine", json.dumps([p.__dict__ for p in self.library.recent("Show", 6, 50)]))
        self.assertNotIn("Zephyrine", [c["name"] for c in self.library.characters("Show", horizon)])

    def test_phrase_location_stops_before_the_passage_with_the_remembered_line(self):
        horizon, found = self.library.locate("Show", "Marlow and Ines argued by the sea")
        self.assertEqual(found.ordinal, 5)
        self.assertEqual(horizon, 4)  # the line may sit in the middle of the passage

    def test_cards_report_when_things_happen(self):
        card = self.library.card("Show", "Ines", self.library.horizon_at_time("Show", 2, 1000))
        self.assertEqual(card["first"]["start"], 0.0)
        self.assertGreater(card["last"]["start"], card["first"]["start"])


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--no-embeddings", *args])
        return code, out.getvalue(), err.getvalue()

    def test_ingest_episodes_in_natural_order_and_ask_by_minute(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for number in (10, 2, 1):
                cues = [(i * 20.0, i * 20.0 + 10.0, f"Episode {number} line {i} about the {'dragon' if number == 10 else 'castle'} "
                         "by the cold grey sea at night, said the old keeper.") for i in range(4)]
                (folder / f"show.S01E{number:02d}.srt").write_text(srt(cues), encoding="utf-8")
            db = str(folder / "lib.db")
            code, out, err = self.run_cli("--db", db, "ingest", *map(str, folder.glob("*.srt")), "--title", "Show")
            self.assertEqual(code, 0, err)
            self.assertIn("3 episodes", out)
            self.assertLess(err.index("S01E01"), err.index("S01E02"))
            self.assertLess(err.index("S01E02"), err.index("S01E10"))
            code, out, err = self.run_cli("--db", db, "ask", "Show", "dragon", "--episode", "2", "--at", "0:30")
            self.assertEqual(code, 0, err)
            self.assertNotIn("dragon", out.lower().replace("nothing relevant", ""))  # episode 10 is far ahead
            code, out, err = self.run_cli("--db", db, "ask", "Show", "castle", "--episode", "2", "--at", "0:30")
            self.assertEqual(code, 0, err)
            self.assertIn("Episode 1 line", out)
            self.assertNotIn("Episode 10", out)
            code, out, err = self.run_cli("--db", db, "ask", "Show", "castle", "--at", "0:30")
            self.assertEqual(code, 1)
            self.assertIn("--at needs --episode", err)
            code, _, err = self.run_cli("--db", db, "ask", "Show", "castle", "--chapter", "1", "--at", "1:00", "--episode", "1")
            self.assertEqual(code, 1)
            code, _, err = self.run_cli("--db", db, "ask", "Show", "castle", "--episode", "1", "--at", "banana")
            self.assertEqual(code, 1)
            self.assertIn("Cannot read", err)


class WebSubtitleTests(WebCase):
    """Reuses the web test fixture (server, token, isolated config)."""

    def episodes(self, secret="ZEPHYRINE"):
        pad = " The evening wind carried the smell of salt across the grey harbour tonight."
        one = srt([(0, 10, "Captain Marlow reached the lighthouse." + pad), (20, 30, "Marlow met the keeper Ines." + pad)])
        two = srt([(0, 10, "Ines spoke to Marlow again." + pad), (20, 30, f"The traitor was {secret}, said Ines." + pad)])
        return [{"name": "show.S01E02.srt", "text": two}, {"name": "show.S01E01.srt", "text": one}]  # sent out of order

    def test_upload_several_subtitle_files_then_ask_by_episode_and_minute(self):
        status, data = self.api("POST", "/api/ingest-subs", {"title": "Serie", "episodes": self.episodes()})
        self.assertEqual((status, data["kind"], data["chapters"]), (200, "video", 2))
        chapters = self.api("GET", "/api/chapters?title=Serie")[1]
        self.assertEqual([c["title"] for c in chapters], ["S01E01", "S01E02"])  # natural order, not upload order
        books = {b["title"]: b for b in self.api("GET", "/api/state")[1]["books"]}
        self.assertEqual(books["Serie"]["kind"], "video")
        status, early = self.api("POST", "/api/ask", {"title": "Serie", "episode": 2, "time": "0:15", "question": "who is the traitor ZEPHYRINE"})
        self.assertEqual(status, 200)
        self.assertNotIn("ZEPHYRINE", json.dumps(early["passages"]).replace("ZEPHYRINE", "", 0)) if False else None
        self.assertNotIn("ZEPHYRINE".lower(), " ".join(p["text"].lower() for p in early["passages"]))
        self.assertTrue(all(p["start"] is not None for p in early["passages"]))
        self.assertTrue(all((p["chapter"], p["start"]) < (2, 10.0) for p in early["passages"]))  # nothing after minute 0:15
        status, late = self.api("POST", "/api/ask", {"title": "Serie", "episode": 2, "time": "1:00", "question": "who is the traitor"})
        self.assertIn("zephyrine", " ".join(p["text"].lower() for p in late["passages"]))
        self.assertEqual(self.api("POST", "/api/ask", {"title": "Serie", "episode": 2, "time": "abc", "question": "x"})[0], 400)
        self.assertEqual(self.api("POST", "/api/ask", {"title": "Serie", "episode": 9, "time": "1:00", "question": "x"})[0], 400)
        self.assertEqual(self.api("POST", "/api/ask", {"title": "S", "episode": 1, "time": "1:00", "question": "x"})[0], 400)  # a book
        status, characters = self.api("POST", "/api/characters", {"title": "Serie", "episode": 2, "time": "0:15"})
        self.assertNotIn("Zephyrine", [c["name"] for c in characters])

    def test_bad_subtitle_uploads_are_rejected(self):
        for body in ({"title": "T", "episodes": []}, {"title": "T", "episodes": "x"}, {"title": "T", "episodes": [{"name": 1, "text": "x"}]},
                     {"title": "T", "episodes": [{"name": "a.srt", "text": "no cues in here"}]},
                     {"title": "T", "episodes": [{"name": "a.srt", "text": "x"}] * 201}):
            self.assertEqual(self.api("POST", "/api/ingest-subs", body)[0], 400, body)
        self.assertEqual(self.api("POST", "/api/ingest-subs", {"title": "T"}, token=None)[0], 403)


@unittest.skipUnless((DATA / "pg1661.txt").exists(), "needs the cached Sherlock Holmes text")
class SeriesFromARealBookTests(unittest.TestCase):
    """Sherlock Holmes turned into 12 'episodes' of 3-second subtitle cues; no read path may pass the minute."""

    @classmethod
    def setUpClass(cls):
        cls.library = Library(":memory:")
        episodes = []
        for number, chapter in enumerate(read_book(DATA / "pg1661.txt"), 1):
            cues, clock = [], 0.0
            for paragraph in chapter.paragraphs:
                for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
                    if sentence.strip():
                        cues.append(subtitles.Cue(clock, clock + 2.9, sentence.strip()))
                        clock += 3.0
            grouped = subtitles.passages(cues)
            episodes.append(Chapter(f"S01E{number:02d}", [g[0] for g in grouped], [(g[1], g[2]) for g in grouped]))
        cls.library.add_work("Sherlock series", episodes)
        cls.rows = cls.library.db.execute("SELECT ordinal, chapter, end_s, text FROM chunks").fetchall()

    @classmethod
    def tearDownClass(cls):
        cls.library.close()

    @staticmethod
    def words(text):
        return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 5}

    def test_nothing_after_the_minute_reaches_any_read_path(self):
        episode_ends = {}
        for _, chapter, end, _ in self.rows:
            episode_ends[chapter] = max(episode_ends.get(chapter, 0), end)
        positions = [(1, 60.0), (1, 400.0), (3, 30.0), (3, 500.0), (6, 120.0), (8, 700.0), (11, 5.0), (12, 100.0)]
        for episode, seconds in positions:
            horizon = self.library.horizon_at_time("Sherlock series", episode, seconds)
            seen = {w for o, ch, end, text in self.rows if o <= horizon for w in self.words(text)}
            self.assertTrue(all(end <= seconds for o, ch, end, _ in self.rows if o <= horizon and ch == episode))
            future = {w for o, ch, end, text in self.rows if o > horizon for w in self.words(text)} - seen
            self.assertTrue(future)
            query = " ".join(sorted(future)[:: max(1, len(future) // 40)])
            outputs = [p.text for p in self.library.search("Sherlock series", query, horizon, limit=50)]
            outputs += [p.text for p in self.library.recent("Sherlock series", horizon, 200)]
            people = self.library.characters("Sherlock series", horizon)
            outputs += list(strings(people))
            for person in people[:6]:
                outputs += list(strings(self.library.card("Sherlock series", person["name"], horizon)))
            leaked = {w for text in outputs for w in self.words(text)} & future
            self.assertFalse(leaked, f"episode {episode} at {seconds}s: {sorted(leaked)[:5]}")


if __name__ == "__main__":
    unittest.main()
