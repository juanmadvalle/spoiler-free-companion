"""Recognise questions that can only be answered with what the reader has not read yet."""

import re

# Deliberately narrow: "who dies?" may be about the past, so it is left to the horizon filter.
_FUTURE = [re.compile(p, re.I) for p in (
    # English
    r"\bhow (?:does|will|did) (?:it|this|the (?:book|story|novel|film|movie|series|show|episode|chapter)|they|he|she) (?:all )?(?:end|finish|conclude|turn out)\b",
    r"\bthe (?:ending|finale)\b",
    r"\b(?:final chapter|last chapter|last page)\b",
    r"\bwhat (?:will|is going to|'s going to|would) happen\b",
    r"\bwhat happens (?:next|later|in the end|at the end|eventually)\b",
    r"\b(?:will|is going to|gonna) (?:he|she|they|it|[a-z]+) (?:die|survive|win|lose|escape|marry|betray|find out|get caught)\b",
    r"\b(?:spoil(?:er|ers)?|tell me what comes)\b",
    # Spanish
    r"\bc[oó]mo (?:acaba|termina|acabar[aá]|terminar[aá]|se resuelve)\b",
    r"\b(?:desenlace|[uú]ltimo cap[ií]tulo|[uú]ltima p[aá]gina)\b",
    r"\bqu[eé] (?:pasar[aá]|va a pasar|ocurrir[aá]|suceder[aá])\b",
    r"\bqu[eé] pasa (?:despu[eé]s|luego|al final|m[aá]s adelante)\b",
    r"\b(?:morir[aá]|sobrevivir[aá]|se casar[aá]|escapar[aá]|traicionar[aá])\b",
    r"\b(?:spoiler|destripa|cu[eé]ntame (?:el )?final)\b",
)]

REFUSAL = ("That depends on what comes after your current position, so it is not established up to this "
           "point (and answering would be a spoiler). Ask about what you have already read.")


def is_future_question(question: str) -> bool:
    return any(pattern.search(question) for pattern in _FUTURE)
