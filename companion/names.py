"""Find character names and expand queries with aliases, using only what the reader has read.

Each full name ("sherlock holmes") remembers the passage where it became established (its second appearance).
A query word is expanded to the other word of its name only if, at the reader's horizon, that name is the one
unambiguous match for both words. Names established later can never influence an earlier answer. Which capitalised
words may be part of a name still uses frequency statistics over the whole book; that is a yes/no decision about a
word and carries no story content (character cards use position-only statistics).
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
        if initial and before.endswith("."):  # "Mr. Watson": the period belongs to the honorific, not a sentence end
            previous = re.search(r"([^\W\d_]+)\.$", before)
            initial = not (previous and previous.group(1).lower() in HONORIFICS)
        word = match.group()
        base = re.sub(r"['’]s?$", "", word.lower()).replace("’", "'")  # "Holmes'" and "Holmes's" are just Holmes
        yield base, word[0].isupper(), initial, match.start(), match.end()


def extract_names(chunks: list[tuple[int, str]]) -> dict[tuple[str, ...], int]:
    """Map each multi-word proper name to the ordinal where it becomes established.

    A name counts as known only from its MIN_NAME_COUNT-th appearance (the second), so the link between its words
    is not available before the passages that justify it.
    """
    capitalized, total = Counter(), Counter()
    for _, text in chunks:
        for word, is_cap, initial, _, _ in _tokens(text):
            if not initial:  # sentence-initial capitals say nothing about proper nouns
                total[word] += 1
                capitalized[word] += is_cap
    proper = {w for w, n in capitalized.items()
              if len(w) >= MIN_TOKEN and w not in _STOP_NAMES and n >= MIN_PROPER_COUNT
              and n / total[w] >= MIN_PROPER_RATIO}
    seen_at = defaultdict(list)
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
                seen_at[name].append(ordinal)
            run, last_end = ([word], end) if word and is_cap and word in proper else ([], None)
    return {name: ordinals[MIN_NAME_COUNT - 1] for name, ordinals in seen_at.items() if len(ordinals) >= MIN_NAME_COUNT}


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


_SPEECH = frozenset(
    "said asked replied cried answered whispered shouted exclaimed murmured remarked observed returned added "
    "continued laughed sighed muttered declared "
    "dijo preguntó respondió contestó exclamó gritó murmuró replicó añadió dice pregunta responde exclama "
    "respondiéndole habló rió suspiró".split())
_CONTRACTION = re.compile(r"'[^\W\d_]{1,2}$")  # I'll, don't, we're: "D'Artagnan" and "O'Brien" are longer and stay
CARD_MIN_MENTIONS = 2  # capitalised mid-sentence mentions before a name gets a card
_COMMON_TITLES = HONORIFICS | {"the", "el", "la"}


def index_mentions(chunks: list[tuple[int, str]]) -> list[tuple[str, int, int, int, int, int]]:
    """Per passage, how each capitalised-somewhere word occurs: (word, ordinal, cap_mid, tot_mid, occurrences, cues).

    cap_mid / tot_mid count mid-sentence occurrences (capitalised / all); a cue is a speech verb right before or
    after the word, or an honorific right before it. The reader's position decides later, at query time, which
    words count as names, so nothing here depends on text the reader has not reached.
    """
    seen_capitalised = set()
    parsed = []
    for ordinal, text in chunks:
        tokens = list(_tokens(text))
        parsed.append((ordinal, tokens))
        for word, is_cap, initial, _, _ in tokens:
            if (is_cap and not initial and len(word) >= MIN_TOKEN and word not in _STOP_NAMES
                    and word not in HONORIFICS and not _CONTRACTION.search(word)):
                seen_capitalised.add(word)
    rows = []
    for ordinal, tokens in parsed:
        stats = {}
        for i, (word, is_cap, initial, _, _) in enumerate(tokens):
            if word not in seen_capitalised:
                continue
            cap_mid, tot_mid, occ, cues = stats.get(word, (0, 0, 0, 0))
            near_before = tokens[i - 1][0] if i else ""
            near_after = [t[0] for t in tokens[i + 1:i + 3]]
            cue = near_before in HONORIFICS or near_before in _SPEECH or any(w in _SPEECH for w in near_after)
            stats[word] = (cap_mid + (is_cap and not initial), tot_mid + (not initial), occ + 1, cues + bool(cue))
        rows.extend((word, ordinal, *values) for word, values in stats.items())
    return rows
