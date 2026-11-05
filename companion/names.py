"""Find character names and expand queries with aliases, using only what the reader has read.

Each full name ("sherlock holmes") remembers the first passage it appears in. A query word is
expanded to the other word of its name only if, at the reader's horizon, that name is the one
unambiguous match for both words. Names first seen later can never influence an earlier answer.
"""

import re
from collections import Counter, defaultdict

HONORIFICS = frozenset("mr mrs miss ms dr sir lady lord madame mme monsieur colonel captain st "
                       "don doña dona señor señora sr sra srta fray san santa sor doctor".split())
_TOKEN = re.compile(r"[^\W\d_][^\W\d_’']*(?:['’][^\W\d_]+)*")  # letters of any language, accents included
_STOP_NAMES = frozenset("the and but for not you him her his she they them this that what when where who how "
                        "then there here with from have had was were are our your its all any one yes "
                        "los las una unos unas del con por para como pero sus más que cuando donde quien "
                        "este esta ese esa esto eso aquel aquella todo todos nada algo mucho muy".split())
MIN_TOKEN = 3
MIN_PROPER_COUNT = 3
MIN_PROPER_RATIO = 0.8
MIN_NAME_COUNT = 2
MAX_NAME_TOKENS = 3


def _tokens(text: str):
    """Yield (lowercase, is_capitalized, is_sentence_initial, start, end)."""
    for match in _TOKEN.finditer(text):
        before = text[:match.start()].rstrip(" \"“”‘'(«¿¡—–-")  # opening quotes and Spanish dialogue marks
        initial = not before or before[-1] in ".!?:;"
        word = match.group()
        base = re.sub(r"['’]s?$", "", word.lower())  # "Holmes'" and "Holmes's" are just Holmes
        yield base, word[0].isupper(), initial, match.start(), match.end()


def extract_names(chunks: list[tuple[int, str]]) -> dict[tuple[str, ...], int]:
    """Map each multi-word proper name to the ordinal of its first appearance."""
    capitalized, total = Counter(), Counter()
    for _, text in chunks:
        for word, is_cap, initial, _, _ in _tokens(text):
            if not initial:  # sentence-initial capitals say nothing about proper nouns
                total[word] += 1
                capitalized[word] += is_cap
    proper = {w for w, n in capitalized.items()
              if len(w) >= MIN_TOKEN and w not in _STOP_NAMES and n >= MIN_PROPER_COUNT
              and n / total[w] >= MIN_PROPER_RATIO}
    first, count = {}, Counter()
    for ordinal, text in chunks:
        run, last_end = [], None
        for word, is_cap, _, start, end in [*_tokens(text), ("", False, False, -1, -1)]:
            gap = text[last_end:start] if last_end is not None else None
            # A period joins words only after an honorific ("Mr. Darcy"), never across sentences.
            joined = gap == " " or (gap == ". " and run[-1:] and run[-1] in HONORIFICS)
            if word and is_cap and (word in proper or word in HONORIFICS) and (not run or joined):
                run.append(word)
                last_end = end
                continue
            name = tuple(w for w in run if w not in HONORIFICS)
            if 2 <= len(name) <= MAX_NAME_TOKENS:
                count[name] += 1
                first.setdefault(name, ordinal)
            run, last_end = ([word], end) if word and is_cap and word in proper else ([], None)
    return {name: ordinal for name, ordinal in first.items() if count[name] >= MIN_NAME_COUNT}


def expand_words(words: list[str], names: dict[tuple[str, ...], int], horizon: int) -> list[str]:
    """Extra query words from names already seen at the horizon, only when unambiguous."""
    known = defaultdict(list)
    for name, seen in names.items():
        if seen <= horizon:
            for token in name:
                known[token].append(name)
    extra = []
    for word in words:
        matches = known.get(word, [])
        if len(matches) != 1:
            continue
        for other in matches[0]:
            if other != word and len(known[other]) == 1 and other not in words and other not in extra:
                extra.append(other)
    return extra
