"""Compare embedding models on the paraphrase questions of two books (smaller, so each model indexes quickly)."""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_semantic as E  # noqa: E402
from companion.embed import FastEmbedder  # noqa: E402
from companion.store import Library  # noqa: E402

MODELS = sys.argv[1:] or ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "snowflake/snowflake-arctic-embed-m"]
BOOKS = ("Alice", "Frankenstein")

for model in MODELS:
    embedder = FastEmbedder(model)
    started, hits, total, chunks = time.time(), 0, 0, 0
    for book in BOOKS:
        library = Library(E.DATA / f"eval-{book}.db", embedder=embedder)
        before = time.time()
        chunks += library.embed_work(book)
        horizon = library.db.execute("SELECT MAX(ordinal) FROM chunks").fetchone()[0]
        for case_book, question, pattern in E.CASES:
            if case_book == book:
                total += 1
                hits += any(re.search(pattern, p.text, re.I) for p in library.search(book, question, horizon, limit=5))
        library.close()
    print(f"{model:<40} hit@5 {hits}/{total}   newly embedded {chunks} passages, {time.time() - started:.0f}s total", flush=True)
