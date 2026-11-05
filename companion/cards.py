"""Character cards computed on the fly from what the reader has reached, never from later text.

Every number, name and sentence comes from rows and passages with `ordinal <= horizon`: whether a word counts as a
name is decided from its mentions up to the reader's position, so a name that only becomes frequent later does not
appear early and nothing about future events can leak into a card.
"""

import re
import sqlite3
from collections import Counter, defaultdict

from . import lang as langs
from .names import CARD_MIN_MENTIONS, HONORIFICS, MIN_PROPER_RATIO, _STOP_NAMES

MAX_EXCERPT = 320


def _clip(text: str, limit: int = MAX_EXCERPT) -> str:
    """Cut at a word boundary, so no half word is ever shown."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"

_SENTENCE = re.compile(r"(?<=[.!?])[\"”’']?\s+")
_DESCRIBED = {
    # "X is/was a ...", "X, a/the ...", "called X": sentences that say who someone is
    "en": (r"\b{n}\b\s+(?:is|was)\s+(?:a|an|the|one|my|his|her|not|very|so)\b",
           r"\b{n}\b,\s+(?:a|an|the|his|her|my)\s+\w+",
           r"\b(?:called|named)\s+{n}\b"),
    "es": (r"\b{n}\b\s+(?:es|era|fue)\s+(?:un|una|el|la|mi|su|muy|tan|de)\b",
           r"\b{n}\b,\s+(?:un|una|el|la|su|mi)\s+\w+",
           r"\b(?:llamad[oa]|llamaba|llamó)\s+(?:\w+\s+)?{n}\b"),
}


def proper_tokens(db: sqlite3.Connection, work_id: int, horizon: int) -> dict[str, dict]:
    """Words that count as names at this position, with their statistics up to it."""
    rows = db.execute(
        "SELECT token, SUM(cap_mid), SUM(tot_mid), SUM(occ), SUM(cues), MIN(ordinal), MAX(ordinal) FROM mentions "
        "WHERE work_id = ? AND ordinal <= ? GROUP BY token "
        "HAVING SUM(cap_mid) >= ? AND SUM(cap_mid) >= ? * SUM(tot_mid)",
        (work_id, horizon, CARD_MIN_MENTIONS, MIN_PROPER_RATIO)).fetchall()
    return {t: {"mid": cap, "occ": occ, "cues": cues, "first": first, "last": last}
            for t, cap, _, occ, cues, first, last in rows
            if len(t) >= 3 and t not in _STOP_NAMES and t not in HONORIFICS}


def _groups(db: sqlite3.Connection, work_id: int, horizon: int, tokens: dict[str, dict]) -> dict[str, list[str]]:
    """primary token -> its other tokens, from full names seen by the horizon where both words are unambiguous."""
    names = [tuple(n.split()) for (n,) in db.execute(
        "SELECT name FROM names WHERE work_id = ? AND first_seen <= ?", (work_id, horizon))]
    owner = defaultdict(set)
    for name in names:
        for token in name:
            owner[token].add(name)
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for name in names:
        members = [t for t in name if t in tokens and len(owner[t]) == 1]
        for other in members[1:]:
            parent[find(other)] = find(members[0])
    clusters = defaultdict(list)
    for token in tokens:
        clusters[find(token)].append(token)
    result = {}
    for members in clusters.values():
        primary = max(members, key=lambda t: tokens[t]["occ"])
        result[primary] = sorted(m for m in members if m != primary)
    return result


def _display(token: str) -> str:
    return token[:1].upper() + token[1:]


def _sentence_with(text: str, token: str) -> str:
    pattern = re.compile(r"\b" + re.escape(token) + r"\b", re.I)
    for sentence in _SENTENCE.split(" ".join(text.split())):
        if pattern.search(sentence):
            return _clip(sentence)
    return _clip(" ".join(text.split()))


def list_characters(db: sqlite3.Connection, work_id: int, horizon: int, limit: int = 40) -> list[dict]:
    """The names met so far, most mentioned first; words that are merely aliases of another entry are folded in."""
    tokens = proper_tokens(db, work_id, horizon)
    groups = _groups(db, work_id, horizon, tokens)
    members = {m for others in groups.values() for m in others}
    out = []
    for primary, others in groups.items():
        info = tokens[primary]
        out.append({"name": _display(primary), "aliases": [_display(o) for o in others],
                    "kind": "character" if info["cues"] else "other", "mentions": info["occ"],
                    "first_ordinal": info["first"], "last_ordinal": info["last"]})
    out = [c for c in out if c["name"].lower() not in members]
    out.sort(key=lambda c: (-c["mentions"], c["first_ordinal"]))
    return out[:limit]


def card(db: sqlite3.Connection, work_id: int, language: str, name: str, horizon: int) -> dict:
    tokens = proper_tokens(db, work_id, horizon)
    groups = _groups(db, work_id, horizon, tokens)
    wanted = [langs.fold(w) for w in langs.words(name)]
    candidates = [t for t in tokens if langs.fold(t) in wanted]
    if not candidates:
        raise LookupError(f"No card for '{name}' yet: that name has not appeared often enough up to your position")
    primary = next((p for p, others in groups.items() if set(candidates) & ({p} | set(others))), None)
    primary = primary or max(candidates, key=lambda t: tokens[t]["occ"])
    members = [primary, *groups.get(primary, [])]

    marks = ",".join("?" * len(members))
    ordinals = sorted({o for (o,) in db.execute(
        f"SELECT DISTINCT ordinal FROM mentions WHERE work_id = ? AND ordinal <= ? AND token IN ({marks})",
        (work_id, horizon, *members))})
    passages = {o: (chapter, title, text) for o, chapter, title, text in db.execute(
        "SELECT ordinal, chapter, chapter_title, text FROM chunks WHERE work_id = ? AND ordinal <= ? AND ordinal IN "
        "(SELECT DISTINCT ordinal FROM mentions WHERE work_id = ? AND ordinal <= ? AND token IN (" + marks + "))",
        (work_id, horizon, work_id, horizon, *members))}

    def point(ordinal: int, token: str) -> dict:
        chapter, title, text = passages[ordinal]
        return {"ordinal": ordinal, "chapter": chapter, "chapter_title": title, "text": _sentence_with(text, token)}

    described, seen_text = [], set()
    patterns = [re.compile(p.format(n=re.escape(t)), re.I) for t in members for p in _DESCRIBED.get(language, _DESCRIBED["en"])]
    for ordinal in ordinals:
        chapter, title, text = passages[ordinal]
        for sentence in _SENTENCE.split(" ".join(text.split())):
            if sentence not in seen_text and any(p.search(sentence) for p in patterns):
                seen_text.add(sentence)
                described.append({"ordinal": ordinal, "chapter": chapter, "chapter_title": title, "text": _clip(sentence)})
        if len(described) >= 3:
            break

    together = Counter()
    wanted_ordinals = set(ordinals)
    for ordinal, token in db.execute("SELECT ordinal, token FROM mentions WHERE work_id = ? AND ordinal <= ?",
                                     (work_id, horizon)):
        if ordinal in wanted_ordinals and token in tokens and token not in members:
            together[token] += 1
    primaries = set(groups)
    related = [{"name": _display(t), "together": n} for t, n in together.most_common(12)
               if n >= 2 and (t in primaries)][:6]

    chapters = sorted({passages[o][0] for o in ordinals})
    info = tokens[primary]
    return {"name": _display(primary), "aliases": [_display(m) for m in members[1:]],
            "kind": "character" if info["cues"] else "other", "mentions": info["occ"], "passages": len(ordinals),
            "chapters": chapters, "first": point(ordinals[0], primary), "last": point(ordinals[-1], primary),
            "described_as": described[:3], "related": related, "position": horizon}
