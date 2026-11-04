"""Does a shorter passage size find paraphrased questions better? Alice and Frankenstein, keyword and hybrid."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_semantic as E  # noqa: E402
from companion import ingest  # noqa: E402
from companion.embed import FastEmbedder  # noqa: E402
from companion.ingest import read_book  # noqa: E402
from companion.store import Library  # noqa: E402

SIZES = [int(a) for a in sys.argv[1:]] or [700, 500, 350, 250]
BOOKS = ("Alice", "Frankenstein")
embedder = FastEmbedder()

for size in SIZES:
    ingest.MAX_CHUNK_CHARS = size
    keyword = hybrid = total = passages = 0
    for book in BOOKS:
        library = Library(E.DATA / f"chunk{size}-{book}.db", embedder=embedder)
        if book not in library.works():
            passages += library.add_work(book, read_book(E.DATA / f"pg{E.BOOKS[book]}.txt"))
        else:
            passages += library.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        library.embed_work(book)
        horizon = library.db.execute("SELECT MAX(ordinal) FROM chunks").fetchone()[0]
        for case_book, question, pattern in E.CASES:
            if case_book != book:
                continue
            total += 1
            library.embedder = None
            keyword += any(re.search(pattern, p.text, re.I) for p in library.search(book, question, horizon, limit=5))
            library.embedder = embedder
            hybrid += any(re.search(pattern, p.text, re.I) for p in library.search(book, question, horizon, limit=5))
        library.close()
    print(f"max {size:>4} chars: {passages:>5} passages   keyword {keyword}/{total}   hybrid {hybrid}/{total}", flush=True)
