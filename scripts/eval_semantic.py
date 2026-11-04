"""Paraphrase benchmark: questions worded differently from the text, judged by a regex on the passage.

Compares keyword-only search with hybrid (keyword + local embeddings) on the cached real books.
Vectors are cached in tests/data/eval.db, so only the first run is slow.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from companion.embed import FastEmbedder, FastReranker  # noqa: E402
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
    # second batch
    ("Alice", "the girl nearly drowns in a pool made of her own weeping", r"pool of tears"),
    ("Alice", "a feline that disappears leaving only its smile floating in the air", r"grin without a cat|Cheshire"),
    ("Alice", "a tea party where it is stuck forever at the same hour", r"six o'clock|Hatter"),
    ("Alice", "playing cards secretly painting white flowers a different colour", r"painting the roses|paint(ed|ing)? them red"),
    ("Pride", "at the ball the wealthy newcomer dances with the eldest Bennet daughter", r"Bingley.{0,200}danced|danced.{0,200}Bingley"),
    ("Pride", "the pompous clergyman cousin asks for her hand and she declines", r"offer of marriage|Collins.{0,200}(proposal|refus)"),
    ("Pride", "he sends her a long written explanation of his behaviour", r"Be not alarmed, madam"),
    ("Sherlock", "a gem is hidden inside a bird that was bought for Christmas", r"crop of the goose|carbuncle"),
    ("Sherlock", "a beggar in the city is actually a respectable man in disguise", r"twisted lip|Hugh Boone"),
    ("Sherlock", "someone is killed by a snake that crawls through an air vent", r"ventilator|bell-rope"),
    ("Frankenstein", "the creature begs his maker to build him a mate", r"create a female|female"),
    ("Frankenstein", "the scientist destroys the half-built partner in a rage", r"tore to pieces"),
    ("Frankenstein", "the creature reads a famous epic poem and compares himself to the first man", r"Paradise Lost"),
    ("Frankenstein", "the bride is found dead on the night of the wedding", r"bridal bed|lifeless and inanimate"),
]


def hits(library: Library, title: str, question: str, pattern: str, k: int) -> bool:
    horizon = library.db.execute("SELECT MAX(ordinal) FROM chunks").fetchone()[0]
    return any(re.search(pattern, p.text, re.I) for p in library.search(title, question, horizon, limit=k))


def main(k: int = 5) -> None:
    embedder, reranker = FastEmbedder(), FastReranker()
    variants = {"keyword": (None, None), "hybrid": (embedder, None), "hybrid+rerank": (embedder, reranker)}
    results = dict.fromkeys(variants, 0)
    for book, question, pattern in CASES:
        db = DATA / f"eval-{book}.db"
        library = Library(db, embedder=embedder)
        if book not in library.works():
            library.add_work(book, read_book(DATA / f"pg{BOOKS[book]}.txt"))
        library.embed_work(book)
        marks = []
        for name, (emb, rer) in variants.items():
            library.embedder, library.reranker = emb, rer
            ok = hits(library, book, question, pattern, k)
            results[name] += ok
            marks.append(f"{name if ok else '-':<14}")
        library.close()
        print(" ".join(marks), f"{book}: {question}")
    n = len(CASES)
    print(f"\nhit@{k} of {n}: " + "  ".join(f"{name} {count}" for name, count in results.items()))


if __name__ == "__main__":
    main()
