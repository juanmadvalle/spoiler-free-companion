"""Known-item retrieval benchmark on real books: can a few remembered words find the passage?

Queries are built from the rarest words of a passage, with some words inflected (plural/-ed/-ing
swaps) the way a reader would phrase them. Reports hit@k with the horizon at the passage itself.
"""
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from companion import store  # noqa: E402
from companion.ingest import chunk, read_book  # noqa: E402

BOOKS = {1661: "Sherlock", 1342: "Pride", 11: "Alice", 84: "Frankenstein"}
STOP = store._STOP | {"said", "would", "could", "there", "their", "which", "about", "other", "these", "those"}


def inflect(word: str, k: int) -> str:
    if k % 3 == 0:
        return word[:-1] if word.endswith("s") else word + "s"
    if k % 3 == 1:
        return word[:-2] if word.endswith("ed") else word + "ed"
    return word[:-3] if word.endswith("ing") else word + "ing"


def run(label: str, k: int = 3, per_book: int = 150) -> dict:
    """Success = one of the top k passages contains all the (uninflected) query words."""
    hits = collections.Counter()
    for book_id, title in BOOKS.items():
        chapters = read_book(Path(__file__).parent.parent / "tests" / "data" / f"pg{book_id}.txt")
        library = store.Library(":memory:")
        library.add_work(title, chapters)
        rows = library.db.execute("SELECT ordinal, text FROM chunks ORDER BY ordinal").fetchall()
        sets = {o: set(store._words(t)) for o, t in rows}
        df = collections.Counter(w for ws in sets.values() for w in ws)
        step = max(1, len(rows) // per_book)
        for ordinal, text in rows[::step][:per_book]:
            words = [w for w in dict.fromkeys(store._words(text))
                     if len(w) >= 5 and w not in STOP and w.isalpha() and 4 <= df[w] <= 60]
            if len(words) < 3:
                continue
            query_words = sorted(words, key=lambda w: df[w])[:3]
            for mode in ("exact", "inflected"):
                query = " ".join(inflect(w, i) if mode == "inflected" else w for i, w in enumerate(query_words))
                found = library.search(title, query, ordinal, limit=k)
                hits[mode, "n"] += 1
                hits[mode, "hit"] += any(set(query_words) <= sets[p.ordinal] for p in found)
        library.close()
    print(f"{label:<22}" + "  ".join(f"{m}: {hits[m, 'hit']}/{hits[m, 'n']} = {hits[m, 'hit'] / hits[m, 'n']:.1%}"
                                   for m in ("exact", "inflected")))
    return hits


if __name__ == "__main__":
    run("current")
