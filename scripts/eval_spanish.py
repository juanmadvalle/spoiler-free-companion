"""Does Spanish support help? Compares the Spanish pipeline with what happened before (Spanish text indexed as English).

A. Known item: three remembered words, each changed the way people type (plural/gender swap, accents dropped).
B. Paraphrase: hand-written questions about Lazarillo de Tormes and Don Quijote, judged by a regex on the passage;
   keyword-only for both books, plus hybrid (multilingual embeddings) for the short book.
"""
import collections
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from companion import lang  # noqa: E402
from companion.embed import LanguageEmbedders  # noqa: E402
from companion.ingest import read_book  # noqa: E402
from companion.store import Library  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "tests" / "data"
BOOKS = {"Lazarillo": 320, "Quijote": 2000}
CASES = [
    ("Lazarillo", "el ciego golpea a Lázaro contra un animal de piedra para enseñarle a ser listo", r"toro|piedra"),
    ("Lazarillo", "el amo que guardaba el pan bajo llave en un baúl", r"arca|llave"),
    ("Lazarillo", "el hidalgo muy pobre que presumía de noble y no tenía qué comer", r"escudero|hidalgo"),
    ("Lazarillo", "el vendedor de indulgencias que engaña al pueblo con falsos milagros", r"buldero|bula"),
    ("Lazarillo", "comen racimos de fruta cogiendo de tres en tres", r"uvas"),
    ("Quijote", "el caballero embiste contra unas aspas enormes creyendo que son gigantes", r"molinos|gigantes"),
    ("Quijote", "el labrador ambicioso que espera mandar una isla", r"ínsula|ínsulas|insula"),
    ("Quijote", "unos condenados que van encadenados y don Quijote los libera", r"galeotes|cadena"),
    ("Quijote", "la palangana de barbero que él cree un casco de oro", r"Mambrino|bacía"),
    ("Quijote", "los animales de lana que toma por soldados de dos ejércitos", r"ovejas|carneros|ejército"),
    ("Quijote", "es armado caballero por el dueño de una posada", r"venta|ventero"),
]


def real_variant(word: str, forms: dict[str, list[str]]) -> str:
    """Another spelling of the same word that the book itself uses (other number, gender or verb form)."""
    others = [w for w in forms.get(lang.stem_es(word), []) if w != word]
    return others[0] if others else word


def build(title: str, language: str | None, embedder=None) -> Library:
    path = DATA / f"es-{title}-{language or 'auto'}.db"
    library = Library(path, embedder=embedder)
    if title not in library.works():
        library.add_work(title, read_book(DATA / f"pg{BOOKS[title]}.txt"), language)
    return library


def known_item(title: str, language: str, per_book: int = 150) -> str:
    """Three remembered words are typed as (exact) the book's spelling, (variant) another form of the same
    word found elsewhere in the book, (no accents) without accents. Success: a top-3 passage has the originals."""
    library = build(title, language)
    rows = library.db.execute("SELECT ordinal, text FROM chunks ORDER BY ordinal").fetchall()
    sets = {o: set(lang.words(t)) for o, t in rows}
    df = collections.Counter(w for ws in sets.values() for w in ws)
    forms = collections.defaultdict(list)
    for word, _ in df.most_common():  # most frequent spelling first
        if word.isalpha() and len(word) >= 4:
            forms[lang.stem_es(word)].append(word)
    hits, changed = collections.Counter(), 0
    step = max(1, len(rows) // per_book)
    for ordinal, text in rows[::step][:per_book]:
        words = [w for w in dict.fromkeys(lang.words(text)) if len(w) >= 5 and w.isalpha() and 3 <= df[w] <= 80]
        if len(words) < 3:
            continue
        target = sorted(words, key=lambda w: df[w])[:3]
        variants = [real_variant(w, forms) for w in target]
        changed += variants != target
        queries = {"exact": target, "variant": variants, "no accents": [lang.fold(w) for w in target]}
        for mode, query_words in queries.items():
            found = library.search(title, " ".join(query_words), ordinal, limit=3)
            hits[mode, "n"] += 1
            hits[mode, "hit"] += any(set(target) <= sets[p.ordinal] for p in found)
    library.close()
    return "  ".join(f"{m}: {hits[m, 'hit']}/{hits[m, 'n']} = {hits[m, 'hit'] / hits[m, 'n']:.0%}"
                     for m in ("exact", "variant", "no accents")) + f"   ({changed} queries had a real variant)"


def paraphrase(language: str | None, embedder=None, only: str | None = None) -> tuple[int, int]:
    done = hits = 0
    for title, question, pattern in CASES:
        if only and title != only:
            continue
        library = build(title, language, embedder)
        if embedder:
            library.embed_work(title)
        horizon = library.db.execute("SELECT MAX(ordinal) FROM chunks").fetchone()[0]
        done += 1
        hits += any(re.search(pattern, p.text, re.I) for p in library.search(title, question, horizon, limit=5))
        library.close()
    return hits, done


if __name__ == "__main__":
    print("A. known-item lookups (top 3), words changed the way people type them")
    for title in BOOKS:
        print(f"  {title:<10} as English (before): {known_item(title, 'en')}")
        print(f"  {title:<10} Spanish pipeline:    {known_item(title, 'es')}", flush=True)
    print("B. paraphrase questions, relevant passage in the top 5")
    for label, language in (("as English (before)", "en"), ("Spanish keyword", "es")):
        hits, total = paraphrase(language)
        print(f"  {label:<22} {hits}/{total}", flush=True)
    started = time.time()
    hits, total = paraphrase("es", LanguageEmbedders(), only="Lazarillo")
    keyword, _ = paraphrase("es", None, only="Lazarillo")
    print(f"  Lazarillo only: Spanish keyword {keyword}/{total}, hybrid with multilingual embeddings {hits}/{total} ({time.time() - started:.0f}s)")
