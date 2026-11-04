"""SQLite store. Every read takes a horizon and filters in SQL, so later text never leaves the database."""

import difflib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .ingest import Chapter, chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS works (id INTEGER PRIMARY KEY, title TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, work_id INTEGER NOT NULL REFERENCES works(id),
    ordinal INTEGER NOT NULL, chapter INTEGER NOT NULL, chapter_title TEXT NOT NULL,
    text TEXT NOT NULL, UNIQUE (work_id, ordinal));
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id');
"""


@dataclass(frozen=True)
class Passage:
    ordinal: int
    chapter: int
    chapter_title: str
    text: str


class Library:
    def __init__(self, path: Path | str):
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def add_work(self, title: str, chapters: list[Chapter]) -> int:
        """Replace any previous copy of the work; ordinals follow reading order."""
        with self.db:
            self._delete_work(title)
            work_id = self.db.execute("INSERT INTO works (title) VALUES (?)", (title,)).lastrowid
            ordinal = 0
            for number, chapter in enumerate(chapters, start=1):
                for text in chunk(chapter.paragraphs):
                    ordinal += 1
                    row = self.db.execute(
                        "INSERT INTO chunks (work_id, ordinal, chapter, chapter_title, text) VALUES (?,?,?,?,?)",
                        (work_id, ordinal, number, chapter.title, text)).lastrowid
                    self.db.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (row, text))
        return ordinal

    def _delete_work(self, title: str) -> None:
        row = self.db.execute("SELECT id FROM works WHERE title = ?", (title,)).fetchone()
        if row:
            for chunk_id, text in self.db.execute("SELECT id, text FROM chunks WHERE work_id = ?", row).fetchall():
                self.db.execute("INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', ?, ?)",
                                (chunk_id, text))
            self.db.execute("DELETE FROM chunks WHERE work_id = ?", row)
            self.db.execute("DELETE FROM works WHERE id = ?", row)

    def _work(self, title: str) -> int:
        row = self.db.execute("SELECT id FROM works WHERE title = ?", (title,)).fetchone()
        if not row:
            raise KeyError(f"Unknown work: {title}")
        return row[0]

    def works(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT title FROM works ORDER BY title")]

    def chapters(self, title: str) -> list[tuple[int, str]]:
        """Chapter list is only titles and numbers, so it carries no story content beyond headings."""
        return self.db.execute(
            "SELECT DISTINCT chapter, chapter_title FROM chunks WHERE work_id = ? ORDER BY chapter",
            (self._work(title),)).fetchall()

    def horizon_after_chapter(self, title: str, chapter: int) -> int:
        """Horizon for 'I have finished this chapter' (inclusive)."""
        row = self.db.execute("SELECT MAX(ordinal) FROM chunks WHERE work_id = ? AND chapter <= ?",
                              (self._work(title), chapter)).fetchone()
        if row[0] is None:
            raise ValueError(f"Chapter {chapter} not found")
        return row[0]

    def locate(self, title: str, phrase: str, min_ratio: float = 0.6) -> tuple[int, Passage]:
        """Find where the reader is from a remembered phrase; returns (horizon, matched passage).

        The search scans the whole book, but only the position is returned for the horizon, and the
        caller shows the matched passage so the reader can confirm it is the right place.
        """
        words = _words(phrase)
        if not words:
            raise ValueError("Phrase has no searchable words")
        rows = self.db.execute(
            "SELECT c.ordinal, c.chapter, c.chapter_title, c.text FROM chunks_fts f "
            "JOIN chunks c ON c.id = f.rowid WHERE chunks_fts MATCH ? AND c.work_id = ? "
            "ORDER BY bm25(chunks_fts) LIMIT 10", (_fts_query(words), self._work(title))).fetchall()
        needle = " ".join(words)
        best, score = None, 0.0
        for row in rows:
            haystack = " ".join(_words(row[3]))
            matcher = difflib.SequenceMatcher(None, needle, haystack, autojunk=False)
            span = matcher.find_longest_match(0, len(needle), 0, len(haystack)).size
            ratio = span / len(needle)
            if ratio > score:
                best, score = row, ratio
        if best is None or score < min_ratio:
            raise LookupError("No close match; try a longer or more exact phrase")
        return best[0], Passage(*best)

    def search(self, title: str, query: str, horizon: int, limit: int = 6) -> list[Passage]:
        """Relevant passages at or before the horizon, in reading order."""
        words = _words(query)
        if not words:
            return []
        rows = self.db.execute(
            "SELECT c.ordinal, c.chapter, c.chapter_title, c.text FROM chunks_fts f "
            "JOIN chunks c ON c.id = f.rowid "
            "WHERE chunks_fts MATCH ? AND c.work_id = ? AND c.ordinal <= ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?", (_fts_query(words), self._work(title), horizon, limit)).fetchall()
        return sorted((Passage(*r) for r in rows), key=lambda p: p.ordinal)

    def recent(self, title: str, horizon: int, limit: int = 12) -> list[Passage]:
        """The last passages up to the horizon, in reading order (for recaps)."""
        rows = self.db.execute(
            "SELECT ordinal, chapter, chapter_title, text FROM chunks "
            "WHERE work_id = ? AND ordinal <= ? ORDER BY ordinal DESC LIMIT ?",
            (self._work(title), horizon, limit)).fetchall()
        return [Passage(*r) for r in reversed(rows)]

    def chapter_passages(self, title: str, chapter: int, horizon: int, limit: int = 8, offset: int = 0) -> list[Passage]:
        """Passages of one chapter, still capped at the horizon even if the chapter is partly unread."""
        rows = self.db.execute(
            "SELECT ordinal, chapter, chapter_title, text FROM chunks "
            "WHERE work_id = ? AND chapter = ? AND ordinal <= ? ORDER BY ordinal LIMIT ? OFFSET ?",
            (self._work(title), chapter, horizon, limit, offset)).fetchall()
        return [Passage(*r) for r in rows]


_STOP = frozenset("a an and are as at be but by for from had has have he her his i in is it its of on or "
                  "she that the their them they this to was were what when where which who why will with you "
                  "me my about did does do how tell".split())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def _fts_query(words: list[str]) -> str:
    """OR of quoted terms so punctuation or operators in user text can't break the query."""
    terms = [w for w in words if w not in _STOP] or words
    return " OR ".join('"' + w.replace('"', "") + '"' for w in dict.fromkeys(terms))
