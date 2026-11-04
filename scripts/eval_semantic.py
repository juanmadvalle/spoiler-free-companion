"""Paraphrase benchmark: questions worded differently from the text, judged by a regex on the passage.

Compares keyword-only search with hybrid (keyword + local embeddings) on the cached real books.
Vectors are cached in tests/data/eval.db, so only the first run is slow.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from companion.embed import FastEmbedder  # noqa: E402
from companion.ingest import read_book  # noqa: E402
from companion.store import Library  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"
BOOKS = {"Alice": 11, "Pride": 1342, "Sherlock": 1661, "Frankenstein": 84}
# (book, question, regex that a relevant passage must match)
CASES = [
    ("Alice", "why does she suddenly become tiny after drinking from the little bottle", r"shut up like a telescope|DRINK ME"),
    ("Alice", "which animal is rushing around worried that he is behind schedule", r"waistcoat-pocket|dear! Oh dear! I shall be too late"),
    ("Alice", "the strange blue creature who puffs on a pipe and asks who she is", r"hookah"),
    ("Alice", "the ruler who angrily orders that people be beheaded", r"Off with (his|her|their) head"),
    ("Alice", "what animals gather to run around in circles to get dry", r"Caucus-race"),
    ("Pride", "the arrogant rich gentleman declares his love and is turned down", r"ardently|against my will"),
    ("Pride", "a mother obsessed with getting her girls wed to wealthy men", r"single man in possession"),
    ("Pride", "the youngest sister disgraces the family by running away with an officer", r"eloped|gone off|run away"),
    ("Pride", "the heroine visits a grand estate while its owner is supposedly absent", r"Pemberley"),
    ("Sherlock", "the detective figures out someone's job just by glancing at their hand", r"tattoo|calluses|swelling"),
    ("Sherlock", "a monarch from abroad wants an embarrassing picture recovered", r"photograph"),
    ("Sherlock", "men with fiery hair are offered easy money for copying a book", r"Encyclop"),
    ("Sherlock", "the victim says something strange about a ribbon as she dies", r"speckled band"),
    ("Frankenstein", "the scientist's creation opens its eyes for the first time", r"dull yellow eye|breathless horror"),
    ("Frankenstein", "the monster learns to speak by secretly spying on a poor family", r"cottage|De Lacey"),
    ("Frankenstein", "an explorer writes letters to his sibling from the frozen north", r"Saville|my dear sister"),
    ("Frankenstein", "the young woman who is falsely accused of murdering a child", r"Justine"),
]


def hits(library: Library, title: str, question: str, pattern: str, k: int) -> bool:
    horizon = library.db.execute("SELECT MAX(ordinal) FROM chunks").fetchone()[0]
    return any(re.search(pattern, p.text, re.I) for p in library.search(title, question, horizon, limit=k))


def main(k: int = 5) -> None:
    embedder = FastEmbedder()
    results = {"keyword": 0, "hybrid": 0}
    for book, question, pattern in CASES:
        title = book
        db = DATA / f"eval-{book}.db"
        library = Library(db, embedder=embedder)
        if title not in library.works():
            library.add_work(title, read_book(DATA / f"pg{BOOKS[book]}.txt"))
        library.embed_work(title)
        hybrid = hits(library, title, question, pattern, k)
        library.embedder = None
        keyword = hits(library, title, question, pattern, k)
        library.close()
        results["keyword"] += keyword
        results["hybrid"] += hybrid
        print(f"{'keyword' if keyword else '   -   '} {'hybrid' if hybrid else '  -   '}  {book}: {question}")
    n = len(CASES)
    print(f"\nhit@{k}: keyword {results['keyword']}/{n}  hybrid {results['hybrid']}/{n}")


if __name__ == "__main__":
    main()
