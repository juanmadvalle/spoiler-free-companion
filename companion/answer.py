"""Answer from horizon-filtered passages only. The model never sees text past the reader's position."""

import os
from typing import Protocol

from . import keystore
from .guard import REFUSAL, is_future_question, nothing_found, refusal
from .store import Library, Passage

SYSTEM = (
    "You are a spoiler-free reading companion. Answer ONLY from the numbered passages provided; they are "
    "everything the reader has read so far. Do not use outside knowledge of the work, do not hint at what "
    "happens later, and do not speculate about the future. If the passages do not contain the answer, say "
    "that it is not established up to this point. Cite passages like [3]. Reply in the user's language."
)
MODEL = os.environ.get("COMPANION_MODEL", "claude-sonnet-5-5")


class Answerer(Protocol):
    def __call__(self, question: str, passages: list[Passage]) -> str: ...


def build_context(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{p.ordinal}] (chapter {p.chapter}: {p.chapter_title})\n{p.text}" for p in passages)


def extractive(question: str, passages: list[Passage]) -> str:
    """No-LLM fallback: show the relevant passages themselves."""
    if not passages:
        return nothing_found(question)
    return "\n\n".join(f"[{p.ordinal}] chapter {p.chapter}: {p.text}" for p in passages)


def claude(question: str, passages: list[Passage]) -> str:
    import anthropic  # optional dependency

    reply = anthropic.Anthropic(api_key=keystore.anthropic_key()).messages.create(
        model=MODEL, max_tokens=800, system=SYSTEM,
        messages=[{"role": "user", "content": f"Passages:\n{build_context(passages)}\n\nQuestion: {question}"}])
    return "".join(block.text for block in reply.content if block.type == "text")


def compatible(question: str, passages: list[Passage]) -> str:
    """Any OpenAI-compatible endpoint (Gemini free tier, Ollama, ...), see llm.py."""
    from . import llm

    return llm.complete(SYSTEM, f"Passages:\n{build_context(passages)}\n\nQuestion: {question}")


def default_answerer() -> Answerer:
    from . import llm

    if keystore.anthropic_key():
        return claude
    return compatible if llm.settings() else extractive


def check_horizon(passages: list[Passage], horizon: int) -> None:
    """Defence in depth: the SQL filter is the guarantee, this fails loudly if it ever regresses."""
    leaked = [p.ordinal for p in passages if p.ordinal > horizon]
    if leaked:
        raise AssertionError(f"Passages beyond the horizon were retrieved: {leaked}")


def ask(library: Library, title: str, question: str, horizon: int, answerer: Answerer | None = None,
        limit: int = 6) -> tuple[str, list[Passage]]:
    if is_future_question(question):  # no search and no model call
        return refusal(question), []
    passages = library.search(title, question, horizon, limit)
    check_horizon(passages, horizon)
    return (answerer or default_answerer())(question, passages), passages


def recap(library: Library, title: str, horizon: int, answerer: Answerer | None = None,
          limit: int = 12) -> tuple[str, list[Passage]]:
    passages = library.recent(title, horizon, limit)
    check_horizon(passages, horizon)
    question = "Give a short recap of the most recent events and who is involved."
    return (answerer or default_answerer())(question, passages), passages
