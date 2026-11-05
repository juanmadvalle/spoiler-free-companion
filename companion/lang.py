"""Language support: detect the language of a book and stem Spanish words (Snowball algorithm, no dependencies).

English keeps using SQLite's porter tokenizer. Other languages are indexed as pre-stemmed text so the same
analyzer runs on passages at ingest time and on questions at query time.
"""

import re
import unicodedata
from functools import lru_cache

SUPPORTED = ("en", "es")

_STOPWORDS = {
    "en": frozenset("the and of to a in that was he it her his with as for had you said i not but they she at on is "
                    "who what how where does did when why are".split()),
    "es": frozenset("de la que el en y los se del las por un para con no una su al lo como más pero sus le ya o este "
                    "sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos "
                    "es son qué quién cómo dónde cuál cuándo está fue".split()),
}
# Function words dropped from Spanish queries (compared accent-folded, so "qué" and "que" are both dropped).
_QUERY_STOP_ES = frozenset(
    "de la que el en y a los se del las por un para con no una su al lo como mas pero sus le ya o este si porque esta "
    "entre cuando muy sin sobre tambien me hasta hay donde quien desde todo nos durante todos uno les ni contra otros "
    "ese eso ante ellos e esto mi antes algunos unos yo otro otras otra el tanto esa estos mucho quienes nada muchos "
    "cual poco ella estar estas algunas algo nosotros mis tu te ti tus ellas os "
    "es son fue fueron era eran ser sido sea han ha he habia esta estan estaba estaban estar "
    "que quien quienes cual cuales cuando cuanto cuantos cuanta cuantas donde como "
    "dime cuentame explicame hablame dice dijo".split())
_WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*|\d+")


def words(text: str) -> list[str]:
    """Lower-case words, accents kept (accent folding happens in the index and in the stemmer)."""
    return [w.replace("’", "'") for w in _WORD.findall(text.lower())]


def detect(texts) -> str:
    """'es' or 'en' from which language's most common function words dominate a sample of the text."""
    scores = dict.fromkeys(SUPPORTED, 0)
    seen = 0
    for text in texts:
        for word in words(text):
            for language, stop in _STOPWORDS.items():
                if word in stop:
                    scores[language] += 1
        scores["es"] += 2 * sum(text.count(c) for c in "¿¡ñ")  # characters English never uses
        seen += len(text)
        if seen > 200_000:
            break
    return "es" if scores["es"] > 1.3 * scores["en"] else "en"


# ---- Spanish Snowball stemmer ------------------------------------------------------------------------------
_VOWELS = "aeiouáéíóúü"


def _regions(word: str) -> tuple[int, int, int]:
    """Start offsets of RV, R1 and R2 as defined by the Snowball Spanish stemmer."""
    def region_after(start: int) -> int:
        for i in range(start + 1, len(word)):
            if word[i] not in _VOWELS and word[i - 1] in _VOWELS:
                return i + 1
        return len(word)

    r1 = region_after(0) if len(word) > 1 else len(word)
    r2 = region_after(r1) if r1 < len(word) else len(word)
    rv = len(word)
    if len(word) >= 2:
        if word[1] not in _VOWELS:
            for i in range(2, len(word)):
                if word[i] in _VOWELS:
                    rv = i + 1
                    break
        elif word[0] in _VOWELS and word[1] in _VOWELS:
            for i in range(2, len(word)):
                if word[i] not in _VOWELS:
                    rv = i + 1
                    break
        else:
            rv = 3 if len(word) >= 3 else len(word)
    return rv, r1, r2


def _longest(word: str, suffixes, start: int = 0) -> str | None:
    """Longest listed suffix that begins at or after `start` (the region limit)."""
    found = [s for s in suffixes if word.endswith(s) and len(word) - len(s) >= start]
    return max(found, key=len) if found else None


_PRONOUNS = ("selas", "selos", "sela", "selo", "las", "les", "los", "nos", "me", "se", "la", "le", "lo")
_PRONOUN_BEFORE = ("iéndo", "ándo", "ár", "ér", "ír", "ando", "iendo", "ar", "er", "ir", "yendo")
_S1_R2 = ("anzas", "anza", "icos", "icas", "ico", "ica", "ismos", "ismo", "ables", "able", "ibles", "ible", "istas",
          "ista", "osos", "osas", "oso", "osa", "amientos", "amiento", "imientos", "imiento", "adoras", "adores",
          "adora", "ador", "aciones", "ación", "antes", "ante", "ancias", "ancia")
_S1_ALL = _S1_R2 + ("logías", "logía", "uciones", "ución", "encias", "encia", "amente", "mente", "idades", "idad",
                    "ivas", "ivos", "iva", "ivo")
_Y_SUFFIXES = ("yeron", "yendo", "yamos", "yais", "ya", "ye", "yan", "yen", "yo", "yó", "yas", "yes")
_V_AR_ER_IR = (
    "arían", "arías", "arán", "arás", "aríais", "aría", "aréis", "aríamos", "aremos", "ará", "aré", "erían", "erías",
    "erán", "erás", "eríais", "ería", "eréis", "eríamos", "eremos", "erá", "eré", "irían", "irías", "irán", "irás",
    "iríais", "iría", "iréis", "iríamos", "iremos", "irá", "iré", "aba", "ada", "ida", "ía", "ara", "iera", "ad", "ed",
    "id", "ase", "iese", "aste", "iste", "an", "aban", "ían", "aran", "ieran", "asen", "iesen", "aron", "ieron", "ado",
    "ido", "ando", "iendo", "ió", "ar", "er", "ir", "as", "abas", "adas", "idas", "ías", "aras", "ieras", "ases",
    "ieses", "ís", "áis", "abais", "íais", "arais", "ierais", "aseis", "ieseis", "asteis", "isteis", "ados", "idos",
    "amos", "ábamos", "íamos", "imos", "áramos", "iéramos", "iésemos", "ásemos")
_V_EN_ES = ("en", "es", "éis", "emos")
_ACCENTS = str.maketrans("áéíóú", "aeiou")


@lru_cache(maxsize=200_000)  # books repeat a small vocabulary many times
def stem_es(word: str) -> str:
    """Snowball Spanish stemmer; input is one lower-case word, accents allowed."""
    word = word.lower()
    if len(word) < 3 or not word.isalpha():
        return word.translate(_ACCENTS)
    rv, r1, r2 = _regions(word)

    # Step 0: attached pronouns, only when they follow a verb ending that lies in RV
    pronoun = _longest(word, _PRONOUNS, rv)
    if pronoun:
        stem = word[: -len(pronoun)]
        before = _longest(stem, _PRONOUN_BEFORE, rv)
        if before in ("iéndo", "ándo", "ár", "ér", "ír"):
            word = stem.translate(_ACCENTS)
        elif before in ("ando", "iendo", "ar", "er", "ir"):
            word = stem
        elif before == "yendo" and stem[:-5].endswith("u"):
            word = stem
    rv, r1, r2 = _regions(word)

    # Step 1: standard suffixes
    original = word
    suffix = _longest(word, _S1_ALL)
    if suffix:
        cut = len(word) - len(suffix)
        if suffix in _S1_R2 and cut >= r2:
            word = word[:cut]
            if suffix in ("adora", "ador", "adoras", "adores", "ación", "aciones", "ante", "antes", "ancia", "ancias"):
                if word.endswith("ic") and len(word) - 2 >= r2:
                    word = word[:-2]
        elif suffix in ("logías", "logía") and cut >= r2:
            word = word[:cut] + "log"
        elif suffix in ("uciones", "ución") and cut >= r2:
            word = word[:cut] + "u"
        elif suffix in ("encias", "encia") and cut >= r2:
            word = word[:cut] + "ente"
        elif suffix == "amente" and cut >= r1:
            word = word[:cut]
            for tail, deeper in (("iv", ("at",)), ("os", ()), ("ic", ()), ("ad", ())):
                if word.endswith(tail) and len(word) - len(tail) >= r2:
                    word = word[: -len(tail)]
                    for inner in deeper:
                        if word.endswith(inner) and len(word) - len(inner) >= r2:
                            word = word[: -len(inner)]
                    break
        elif suffix == "mente" and cut >= r2:
            word = word[:cut]
            for tail in ("ante", "able", "ible"):
                if word.endswith(tail) and len(word) - len(tail) >= r2:
                    word = word[: -len(tail)]
                    break
        elif suffix in ("idades", "idad") and cut >= r2:
            word = word[:cut]
            for tail in ("abil", "ic", "iv"):
                if word.endswith(tail) and len(word) - len(tail) >= r2:
                    word = word[: -len(tail)]
                    break
        elif suffix in ("ivas", "ivos", "iva", "ivo") and cut >= r2:
            word = word[:cut]
            if word.endswith("at") and len(word) - 2 >= r2:
                word = word[:-2]

    if word == original:
        # Step 2a: verb suffixes beginning with y
        suffix = _longest(word, _Y_SUFFIXES, rv)
        if suffix and word[: len(word) - len(suffix)].endswith("u"):
            word = word[: len(word) - len(suffix)]
        else:
            # Step 2b: other verb suffixes
            suffix = _longest(word, _V_AR_ER_IR + _V_EN_ES, rv)
            if suffix:
                word = word[: len(word) - len(suffix)]
                if suffix in _V_EN_ES and word.endswith("gu") and len(word) - 1 >= rv:
                    word = word[:-1]

    # Step 3: residual suffix
    rv, r1, r2 = _regions(word)
    suffix = _longest(word, ("os", "a", "o", "á", "í", "ó", "e", "é"), rv)
    if suffix:
        word = word[: len(word) - len(suffix)]
        if suffix in ("e", "é") and word.endswith("gu") and len(word) - 1 >= rv:
            word = word[:-1]
    return word.translate(_ACCENTS)


@lru_cache(maxsize=200_000)
def fold(word: str) -> str:
    """Lower-case and strip diacritics (ñ is kept as a letter of its own)."""
    word = word.lower().replace("ñ", "\0")
    word = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
    return word.replace("\0", "ñ")


def query_terms(question_words: list[str], language: str) -> list[str]:
    """Index terms for a question: function words dropped, the rest stemmed (Spanish only; English uses porter)."""
    kept = [w for w in question_words if fold(w) not in _QUERY_STOP_ES] or question_words
    return list(dict.fromkeys(stem_es(w) for w in kept))


def stems(text: str, language: str) -> list[str]:
    """Index terms for a text. English is stemmed by SQLite, so only Spanish goes through here."""
    if language == "es":
        out = []
        for w in words(text):
            out.append(stem_es(w))
            plain = fold(w)
            if plain != w:  # also index the unaccented spelling, for questions typed without accents
                out.append(stem_es(plain))
        return out
    return words(text)
