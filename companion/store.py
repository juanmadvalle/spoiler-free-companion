"""SQLite store. Every read takes a horizon and filters in SQL, so later text never leaves the database."""

import difflib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .ingest import Chapter, chunk
from .names import expand_words, extract_names

SCHEMA = """
CREATE TABLE IF NOT EXISTS names (work_id INTEGER NOT NULL, name TEXT NOT NULL, first_seen INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS vectors (
    work_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, model TEXT NOT NULL, vec BLOB NOT NULL,
    PRIMARY KEY (work_id, ordinal, model));
CREATE TABLE IF NOT EXISTS works (id INTEGER PRIMARY KEY, title TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, work_id INTEGER NOT NULL REFERENCES works(id),
    ordinal INTEGER NOT NULL, chapter INTEGER NOT NULL, chapter_title TEXT NOT NULL,
    text TEXT NOT NULL, UNIQUE (work_id, ordinal));
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61 remove_diacritics 2');
"""
CANDIDATE_FACTOR = 5
RERANK_POOL = 40  # how many hybrid candidates the cross-encoder re-orders
KEYWORD_WEIGHT = 0.5  # relative weight of the keyword ranking when fused with the semantic one
MIN_KEYWORD_COVERAGE = 2  # keyword hits must share at least this many query terms to count in the fusion


@dataclass(frozen=True)
class Passage:
    ordinal: int
    chapter: int
    chapter_title: str
    text: str


class Library:
    def __init__(self, path: Path | str, embedder=None, reranker=None):
        self.embedder = embedder  # optional; semantic search is used only for works that have vectors
        self.reranker = reranker  # optional cross-encoder that re-orders the hybrid candidates
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self._upgrade_fts()

    def _upgrade_fts(self) -> None:
        """Libraries built before stemming get their index rebuilt in place."""
        sql = self.db.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'").fetchone()[0]
        if "porter" not in sql:
            with self.db:
                self.db.execute("DROP TABLE chunks_fts")
                self.db.executescript(SCHEMA)
                self.db.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild')")

    def close(self) -> None:
        self.db.close()

    def add_work(self, title: str, chapters: list[Chapter]) -> int:
        """Replace any previous copy of the work; ordinals follow reading order."""
        with self.db:
            self._delete_work(title)
            work_id = self.db.execute("INSERT INTO works (title) VALUES (?)", (title,)).lastrowid
            ordinal, texts = 0, []
            for number, chapter in enumerate(chapters, start=1):
                for text in chunk(chapter.paragraphs):
                    ordinal += 1
                    texts.append((ordinal, text))
                    row = self.db.execute(
                        "INSERT INTO chunks (work_id, ordinal, chapter, chapter_title, text) VALUES (?,?,?,?,?)",
                        (work_id, ordinal, number, chapter.title, text)).lastrowid
                    self.db.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (row, text))
            self.db.executemany("INSERT INTO names (work_id, name, first_seen) VALUES (?,?,?)",
                                [(work_id, " ".join(n), seen) for n, seen in extract_names(texts).items()])
        return ordinal

    def _delete_work(self, title: str) -> None:
        row = self.db.execute("SELECT id FROM works WHERE title = ?", (title,)).fetchone()
        if row:
            for chunk_id, text in self.db.execute("SELECT id, text FROM chunks WHERE work_id = ?", row).fetchall():
                self.db.execute("INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', ?, ?)",
                                (chunk_id, text))
            self.db.execute("DELETE FROM chunks WHERE work_id = ?", row)
            self.db.execute("DELETE FROM names WHERE work_id = ?", row)
            self.db.execute("DELETE FROM vectors WHERE work_id = ?", row)
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
        work_id = self._work(title)
        semantic = self._semantic(work_id, query, horizon, limit * CANDIDATE_FACTOR)
        words += self.aliases(title, words, horizon)
        rows = self.db.execute(
            "SELECT c.ordinal, c.chapter, c.chapter_title, c.text FROM chunks_fts f "
            "JOIN chunks c ON c.id = f.rowid "
            "WHERE chunks_fts MATCH ? AND c.work_id = ? AND c.ordinal <= ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (_fts_query(words), work_id, horizon, limit * CANDIDATE_FACTOR)).fetchall()
        # Passages covering more distinct query terms first; ties keep bm25 order (sort is stable).
        terms = _terms(words)
        rows = sorted(rows, key=lambda r: -_coverage(r[3], terms))
        if semantic:
            pinned, rest = _fuse(rows, semantic, terms)
            if self.reranker and rest:  # candidates are already horizon-filtered; only their order changes
                pool = rest[:RERANK_POOL]
                scores = self.reranker.rerank(query, [r[3] for r in pool])
                rest = [r for _, r in sorted(zip(scores, pool), key=lambda pair: -pair[0])] + rest[RERANK_POOL:]
            rows = pinned + rest
        return sorted((Passage(*r) for r in rows[:limit]), key=lambda p: p.ordinal)

    def embed_work(self, title: str, batch: int = 64, progress=None) -> int:
        """Embed every passage of a work that has no vector yet for the current model."""
        if self.embedder is None:
            raise RuntimeError("No embedder available (pip install fastembed)")
        work_id, model = self._work(title), self.embedder.name
        todo = self.db.execute(
            "SELECT c.ordinal, c.text FROM chunks c WHERE c.work_id = ? AND NOT EXISTS (SELECT 1 FROM vectors v "
            "WHERE v.work_id = c.work_id AND v.ordinal = c.ordinal AND v.model = ?) ORDER BY c.ordinal",
            (work_id, model)).fetchall()
        for start in range(0, len(todo), batch):
            part = todo[start:start + batch]
            vectors = self.embedder.embed([text for _, text in part])
            with self.db:
                self.db.executemany("INSERT OR REPLACE INTO vectors (work_id, ordinal, model, vec) VALUES (?,?,?,?)",
                                    [(work_id, o, model, v.tobytes()) for (o, _), v in zip(part, vectors)])
            if progress:
                progress(min(start + batch, len(todo)), len(todo))
        return len(todo)

    def has_vectors(self, title: str) -> bool:
        if self.embedder is None:
            return False
        return self.db.execute("SELECT 1 FROM vectors WHERE work_id = ? AND model = ? LIMIT 1",
                               (self._work(title), self.embedder.name)).fetchone() is not None

    def _semantic(self, work_id: int, query: str, horizon: int, limit: int) -> list[tuple]:
        """Nearest passages by meaning. Only vectors of passages at or before the horizon are ever loaded."""
        if self.embedder is None:
            return []
        rows = self.db.execute(
            "SELECT c.ordinal, c.chapter, c.chapter_title, c.text, v.vec FROM vectors v "
            "JOIN chunks c ON c.work_id = v.work_id AND c.ordinal = v.ordinal "
            "WHERE v.work_id = ? AND v.model = ? AND c.ordinal <= ?",
            (work_id, self.embedder.name, horizon)).fetchall()
        if not rows:
            return []
        import numpy as np

        matrix = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        embed_query = getattr(self.embedder, "embed_query", self.embedder.embed)
        scores = matrix @ embed_query([query])[0]
        best = np.argsort(-scores)[:limit]
        return [rows[i][:4] for i in best]

    def aliases(self, title: str, words: list[str], horizon: int) -> list[str]:
        """Other name parts for the query words, from names first seen at or before the horizon."""
        rows = self.db.execute("SELECT name, first_seen FROM names WHERE work_id = ? AND first_seen <= ?",
                               (self._work(title), horizon)).fetchall()
        return expand_words(words, {tuple(name.split()): seen for name, seen in rows}, horizon)

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


def _fuse(keyword: list[tuple], semantic: list[tuple], terms: list[str], k: int = 60) -> tuple[list, list]:
    """Merge keyword and semantic rankings (rows are keyed by ordinal); returns (pinned, rest).

    Passages that contain nearly all the query words are exact lookups and stay first; the rest are
    ordered by weighted reciprocal rank fusion, so paraphrased questions are carried by the semantic side.
    """
    needed = max(2, len(terms) - len(terms) // 5)
    pinned = [row for row in keyword if _coverage(row[3], terms) >= min(needed, len(terms))]
    fused_keyword = [row for row in keyword if _coverage(row[3], terms) >= min(MIN_KEYWORD_COVERAGE, len(terms))]
    score, by_ordinal = {}, {}
    for ranking, weight in ((fused_keyword, KEYWORD_WEIGHT), (semantic, 1.0)):
        for rank, row in enumerate(ranking):
            score[row[0]] = score.get(row[0], 0.0) + weight / (k + rank)
            by_ordinal[row[0]] = row
    pinned_ids = {row[0] for row in pinned}
    rest = [by_ordinal[o] for o in sorted(score, key=lambda o: -score[o]) if o not in pinned_ids]
    return pinned, rest


def _terms(words: list[str]) -> list[str]:
    return list(dict.fromkeys([w for w in words if w not in _STOP] or words))


def _coverage(text: str, terms: list[str]) -> int:
    """How many distinct query terms the passage contains (shared 5-letter prefix counts as a match)."""
    passage = set(_words(text))
    return sum(t in passage or (len(t) >= 5 and any(w.startswith(t[:5]) for w in passage)) for t in terms)


def _fts_query(words: list[str]) -> str:
    """OR of quoted terms so punctuation or operators in user text can't break the query."""
    return " OR ".join('"' + w.replace('"', "") + '"' for w in _terms(words))
