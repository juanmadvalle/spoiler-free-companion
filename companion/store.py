"""SQLite store. Every read takes a horizon and filters in SQL, so later text never leaves the database."""

import difflib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import cards
from . import lang as langs
from .ingest import Chapter, chunk
from .names import expand_words, extract_names, index_mentions

SCHEMA = """
CREATE TABLE IF NOT EXISTS names (work_id INTEGER NOT NULL, name TEXT NOT NULL, first_seen INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS mentions (
    work_id INTEGER NOT NULL, token TEXT NOT NULL, ordinal INTEGER NOT NULL,
    cap_mid INTEGER NOT NULL, tot_mid INTEGER NOT NULL, occ INTEGER NOT NULL, cues INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS mentions_by_token ON mentions (work_id, token, ordinal);
CREATE INDEX IF NOT EXISTS mentions_by_ordinal ON mentions (work_id, ordinal);
CREATE TABLE IF NOT EXISTS vectors (
    work_id INTEGER NOT NULL, ordinal INTEGER NOT NULL, model TEXT NOT NULL, vec BLOB NOT NULL,
    PRIMARY KEY (work_id, ordinal, model));
CREATE TABLE IF NOT EXISTS works (id INTEGER PRIMARY KEY, title TEXT UNIQUE NOT NULL, lang TEXT NOT NULL DEFAULT 'en');
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, work_id INTEGER NOT NULL REFERENCES works(id),
    ordinal INTEGER NOT NULL, chapter INTEGER NOT NULL, chapter_title TEXT NOT NULL,
    text TEXT NOT NULL, start_s REAL, end_s REAL, UNIQUE (work_id, ordinal));
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61 remove_diacritics 2');
-- Languages other than English are indexed as pre-stemmed text (see lang.py), one row per passage.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_x USING fts5(stems, tokenize='unicode61 remove_diacritics 2');
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
        if "lang" not in [row[1] for row in self.db.execute("PRAGMA table_info(works)")]:
            with self.db:  # libraries made before language support are all English
                self.db.execute("ALTER TABLE works ADD COLUMN lang TEXT NOT NULL DEFAULT 'en'")
        if "start_s" not in [row[1] for row in self.db.execute("PRAGMA table_info(chunks)")]:
            with self.db:  # libraries made before subtitle support have no timestamps
                self.db.execute("ALTER TABLE chunks ADD COLUMN start_s REAL")
                self.db.execute("ALTER TABLE chunks ADD COLUMN end_s REAL")
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

    def add_work(self, title: str, chapters: list[Chapter], language: str | None = None) -> int:
        """Replace any previous copy of the work; ordinals follow reading order.

        The language ('en' or 'es') is detected from the text unless given.
        """
        pieces = []  # (chapter number, chapter title, text, start seconds, end seconds)
        for number, chapter in enumerate(chapters, start=1):
            if chapter.times is not None:  # subtitles arrive already grouped, each passage with its time span
                pieces += [(number, chapter.title, text, start, end)
                           for text, (start, end) in zip(chapter.paragraphs, chapter.times)]
            else:
                pieces += [(number, chapter.title, text, None, None) for text in chunk(chapter.paragraphs)]
        language = language if language in langs.SUPPORTED else langs.detect(p[2] for p in pieces)
        with self.db:
            self._delete_work(title)
            work_id = self.db.execute("INSERT INTO works (title, lang) VALUES (?, ?)", (title, language)).lastrowid
            texts = []
            for ordinal, (number, chapter_title, text, start, end) in enumerate(pieces, start=1):
                texts.append((ordinal, text))
                row = self.db.execute(
                    "INSERT INTO chunks (work_id, ordinal, chapter, chapter_title, text, start_s, end_s) "
                    "VALUES (?,?,?,?,?,?,?)", (work_id, ordinal, number, chapter_title, text, start, end)).lastrowid
                if language == "en":
                    self.db.execute("INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)", (row, text))
                else:
                    self.db.execute("INSERT INTO chunks_fts_x (rowid, stems) VALUES (?, ?)",
                                    (row, " ".join(langs.stems(text, language))))
            self.db.executemany("INSERT INTO names (work_id, name, first_seen) VALUES (?,?,?)",
                                [(work_id, " ".join(n), seen) for n, seen in extract_names(texts).items()])
            self._insert_mentions(work_id, texts)
        return len(pieces)

    def _insert_mentions(self, work_id: int, texts: list[tuple[int, str]]) -> None:
        self.db.executemany("INSERT INTO mentions (work_id, token, ordinal, cap_mid, tot_mid, occ, cues) "
                            "VALUES (?,?,?,?,?,?,?)", [(work_id, *row) for row in index_mentions(texts)])

    def _ensure_mentions(self, work_id: int) -> None:
        """Books added before character cards existed get their mention index built on first use."""
        if self.db.execute("SELECT 1 FROM mentions WHERE work_id = ? LIMIT 1", (work_id,)).fetchone():
            return
        texts = self.db.execute("SELECT ordinal, text FROM chunks WHERE work_id = ? ORDER BY ordinal", (work_id,)).fetchall()
        if texts:
            with self.db:
                self._insert_mentions(work_id, texts)

    def characters(self, title: str, horizon: int, limit: int = 40) -> list[dict]:
        """Names met so far (at or before the horizon), most mentioned first."""
        work_id = self._work(title)
        self._ensure_mentions(work_id)
        return cards.list_characters(self.db, work_id, horizon, limit)

    def card(self, title: str, name: str, horizon: int) -> dict:
        """What is known about one name using only passages at or before the horizon."""
        work_id = self._work(title)
        self._ensure_mentions(work_id)
        return cards.card(self.db, work_id, self.lang_of(title), name, horizon)

    def is_timed(self, title: str) -> bool:
        """True for films and series added from subtitles: positions can then be given as episode + minute."""
        return self.db.execute("SELECT 1 FROM chunks WHERE work_id = ? AND start_s IS NOT NULL LIMIT 1",
                               (self._work(title),)).fetchone() is not None

    def times(self, title: str, ordinals: list[int]) -> dict[int, tuple[float, float]]:
        """(start, end) seconds of timed passages; empty for books."""
        if not ordinals:
            return {}
        marks = ",".join("?" * len(ordinals))
        return {o: (a, b) for o, a, b in self.db.execute(
            f"SELECT ordinal, start_s, end_s FROM chunks WHERE work_id = ? AND start_s IS NOT NULL AND ordinal IN ({marks})",
            (self._work(title), *ordinals))}

    def horizon_at_time(self, title: str, episode: int, seconds: float) -> int:
        """Position 'in episode N at this minute': every passage that has fully ended by then, and earlier episodes.

        A passage still being spoken at that minute is not counted, so nothing the viewer has not heard yet leaks.
        """
        work_id = self._work(title)
        if not self.is_timed(title):
            raise ValueError("This title has no timestamps; add subtitles (.srt/.vtt) to use minutes")
        last = self.db.execute("SELECT MAX(chapter) FROM chunks WHERE work_id = ?", (work_id,)).fetchone()[0]
        if not 1 <= episode <= last:
            raise ValueError(f"Episode {episode} not found (this title has episodes 1-{last})")
        within = self.db.execute("SELECT MAX(ordinal) FROM chunks WHERE work_id = ? AND chapter = ? AND end_s <= ?",
                                 (work_id, episode, seconds)).fetchone()[0]
        if within is not None:
            return within
        return self.db.execute("SELECT COALESCE(MAX(ordinal), 0) FROM chunks WHERE work_id = ? AND chapter < ?",
                               (work_id, episode)).fetchone()[0]

    def lang_of(self, title: str) -> str:
        return self.db.execute("SELECT lang FROM works WHERE id = ?", (self._work(title),)).fetchone()[0]

    def _delete_work(self, title: str) -> None:
        row = self.db.execute("SELECT id, lang FROM works WHERE title = ?", (title,)).fetchone()
        if row:
            row, language = (row[0],), row[1]
            for chunk_id, text in self.db.execute("SELECT id, text FROM chunks WHERE work_id = ?", row).fetchall():
                if language == "en":
                    self.db.execute("INSERT INTO chunks_fts (chunks_fts, rowid, text) VALUES ('delete', ?, ?)",
                                    (chunk_id, text))
                else:
                    self.db.execute("DELETE FROM chunks_fts_x WHERE rowid = ?", (chunk_id,))
            self.db.execute("DELETE FROM chunks WHERE work_id = ?", row)
            self.db.execute("DELETE FROM names WHERE work_id = ?", row)
            self.db.execute("DELETE FROM mentions WHERE work_id = ?", row)
            self.db.execute("DELETE FROM vectors WHERE work_id = ?", row)
            self.db.execute("DELETE FROM works WHERE id = ?", row)

    def _work(self, title: str) -> int:
        row = self.db.execute("SELECT id FROM works WHERE title = ?", (title,)).fetchone()
        if not row:
            raise KeyError(f"Unknown work: {title}")
        return row[0]

    def remove_work(self, title: str) -> None:
        with self.db:
            self._work(title)  # KeyError when unknown
            self._delete_work(title)

    def works(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT title FROM works ORDER BY title")]

    def chapters(self, title: str) -> list[tuple[int, str]]:
        """Chapter list is only titles and numbers, so it carries no story content beyond headings."""
        return self.db.execute(
            "SELECT DISTINCT chapter, chapter_title FROM chunks WHERE work_id = ? ORDER BY chapter",
            (self._work(title),)).fetchall()

    def horizon_after_chapter(self, title: str, chapter: int) -> int:
        """Horizon for 'I have finished this chapter' (inclusive)."""
        work_id = self._work(title)
        last = self.db.execute("SELECT MAX(chapter) FROM chunks WHERE work_id = ?", (work_id,)).fetchone()[0]
        if last is None or not 1 <= chapter <= last:  # a typo like 30 for 3 must not mark the whole book as read
            raise ValueError(f"Chapter {chapter} not found (this book has chapters 1-{last})")
        return self.db.execute("SELECT MAX(ordinal) FROM chunks WHERE work_id = ? AND chapter <= ?",
                               (work_id, chapter)).fetchone()[0]

    def locate(self, title: str, phrase: str, min_ratio: float = 0.6) -> tuple[int, Passage]:
        """Find where the reader is from a remembered phrase; returns (horizon, matched passage).

        The search scans the whole book, but only the position is returned for the horizon, and the
        caller shows the matched passage so the reader can confirm it is the right place.
        """
        words = _words(phrase)
        if not words:
            raise ValueError("Phrase has no searchable words")
        work_id = self._work(title)
        table, match = self._fts(self.lang_of(title), words)
        rows = self.db.execute(
            "SELECT c.ordinal, c.chapter, c.chapter_title, c.text FROM " + table + " f "
            "JOIN chunks c ON c.id = f.rowid WHERE " + table + " MATCH ? AND c.work_id = ? "
            "ORDER BY bm25(" + table + ") LIMIT 10", (match, work_id)).fetchall()
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
        if self.is_timed(title):  # the remembered line may sit in the middle of a passage: stop before it
            return max(best[0] - 1, 0), Passage(*best)
        return best[0], Passage(*best)

    def search(self, title: str, query: str, horizon: int, limit: int = 6) -> list[Passage]:
        """Relevant passages at or before the horizon, in reading order."""
        words = _words(query)
        if not words:
            return []
        work_id = self._work(title)
        language = self.lang_of(title)
        semantic = self._semantic(work_id, query, horizon, limit * CANDIDATE_FACTOR)
        words += self.aliases(title, words, horizon)
        table, match = self._fts(language, words)
        rows = self.db.execute(
            "SELECT c.ordinal, c.chapter, c.chapter_title, c.text FROM " + table + " f "
            "JOIN chunks c ON c.id = f.rowid "
            "WHERE " + table + " MATCH ? AND c.work_id = ? AND c.ordinal <= ? "
            "ORDER BY bm25(" + table + ") LIMIT ?",
            (match, work_id, horizon, limit * CANDIDATE_FACTOR)).fetchall()
        # Passages covering more distinct query terms first; ties keep bm25 order (sort is stable).
        cover, count = _coverage_for(language, words)
        rows = sorted(rows, key=lambda r: -cover(r[3]))
        if semantic:
            pinned, rest = _fuse(rows, semantic, cover, count)
            if self.reranker and rest and language == "en":  # candidates are horizon-filtered; only order changes
                pool = rest[:RERANK_POOL]
                scores = self.reranker.rerank(query, [r[3] for r in pool])
                rest = [r for _, r in sorted(zip(scores, pool), key=lambda pair: -pair[0])] + rest[RERANK_POOL:]
            rows = pinned + rest
        return sorted((Passage(*r) for r in rows[:limit]), key=lambda p: p.ordinal)

    @staticmethod
    def _fts(language: str, words: list[str]) -> tuple[str, str]:
        """(FTS table, MATCH expression) for a language; English relies on SQLite's porter stemmer."""
        if language == "en":
            return "chunks_fts", _fts_query(words)
        terms = langs.query_terms(words, language)
        return "chunks_fts_x", " OR ".join('"' + t.replace('"', "") + '"' for t in terms) or '""'

    def _embedder_for(self, work_id: int):
        """The embedder for a work's language (a LanguageEmbedders object picks one; a plain embedder serves all)."""
        if self.embedder is None:
            return None
        language = self.db.execute("SELECT lang FROM works WHERE id = ?", (work_id,)).fetchone()[0]
        pick = getattr(self.embedder, "for_lang", None)
        return pick(language) if pick else self.embedder

    def embed_work(self, title: str, batch: int = 64, progress=None) -> int:
        """Embed every passage of a work that has no vector yet for the current model."""
        if self.embedder is None:
            raise RuntimeError("No embedder available (pip install fastembed)")
        work_id = self._work(title)
        embedder = self._embedder_for(work_id)
        model = embedder.name
        todo = self.db.execute(
            "SELECT c.ordinal, c.text FROM chunks c WHERE c.work_id = ? AND NOT EXISTS (SELECT 1 FROM vectors v "
            "WHERE v.work_id = c.work_id AND v.ordinal = c.ordinal AND v.model = ?) ORDER BY c.ordinal",
            (work_id, model)).fetchall()
        for start in range(0, len(todo), batch):
            part = todo[start:start + batch]
            vectors = embedder.embed([text for _, text in part])
            with self.db:
                self.db.executemany("INSERT OR REPLACE INTO vectors (work_id, ordinal, model, vec) VALUES (?,?,?,?)",
                                    [(work_id, o, model, v.tobytes()) for (o, _), v in zip(part, vectors)])
            if progress:
                progress(min(start + batch, len(todo)), len(todo))
        return len(todo)

    def has_vectors(self, title: str) -> bool:
        if self.embedder is None:
            return False
        work_id = self._work(title)
        return self.db.execute("SELECT 1 FROM vectors WHERE work_id = ? AND model = ? LIMIT 1",
                               (work_id, self._embedder_for(work_id).name)).fetchone() is not None

    def _semantic(self, work_id: int, query: str, horizon: int, limit: int) -> list[tuple]:
        """Nearest passages by meaning. Only vectors of passages at or before the horizon are ever loaded."""
        embedder = self._embedder_for(work_id)
        if embedder is None:
            return []
        rows = self.db.execute(
            "SELECT c.ordinal, c.chapter, c.chapter_title, c.text, v.vec FROM vectors v "
            "JOIN chunks c ON c.work_id = v.work_id AND c.ordinal = v.ordinal "
            "WHERE v.work_id = ? AND v.model = ? AND c.ordinal <= ?",
            (work_id, embedder.name, horizon)).fetchall()
        if not rows:
            return []
        import numpy as np

        matrix = np.frombuffer(b"".join(r[4] for r in rows), dtype=np.float32).reshape(len(rows), -1)
        embed_query = getattr(embedder, "embed_query", embedder.embed)
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


_words = langs.words


def _coverage_for(language: str, words: list[str]):
    """(function giving how many distinct query terms a text contains, number of query terms)."""
    if language == "en":
        terms = _terms(words)
        return (lambda text: _coverage(text, terms)), len(terms)
    terms = set(langs.query_terms(words, language))
    return (lambda text: len(terms & set(langs.stems(text, language)))), len(terms)


def _fuse(keyword: list[tuple], semantic: list[tuple], cover, count: int, k: int = 60) -> tuple[list, list]:
    """Merge keyword and semantic rankings (rows are keyed by ordinal); returns (pinned, rest).

    Passages that contain nearly all the query words are exact lookups and stay first; the rest are
    ordered by weighted reciprocal rank fusion, so paraphrased questions are carried by the semantic side.
    """
    needed = max(2, count - count // 5)
    pinned = [row for row in keyword if cover(row[3]) >= min(needed, count)]
    fused_keyword = [row for row in keyword if cover(row[3]) >= min(MIN_KEYWORD_COVERAGE, count)]
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
