"""Command line: ingest a book, say where you are, ask or recap."""

import argparse
import os
import sys
from pathlib import Path

from . import keystore
from .agent import Session, run_agent
from .answer import ask, recap
from .embed import default_embedder, default_reranker
from .ingest import read_book
from .store import Library

DEFAULT_DB = Path.home() / ".spoiler-free-companion.db"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="companion", description="Context for what you've read, without spoilers.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"Library file (default {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("ingest", help="Add an EPUB or .txt you own (processed locally)")
    add.add_argument("file", type=Path)
    add.add_argument("--title", help="Defaults to the file name")
    add.add_argument("--embed", action="store_true", help="Also build semantic vectors (needs: pip install fastembed)")
    embed = sub.add_parser("embed", help="Build semantic vectors for a book already ingested")
    embed.add_argument("title")
    parser.add_argument("--no-embeddings", action="store_true", help="Keyword search only, even if vectors exist")
    parser.add_argument("--rerank", action="store_true", help="Re-order hybrid candidates with a cross-encoder (slower; no measured gain yet)")

    sub.add_parser("list", help="List books")
    sub.add_parser("models", help="List models offered by the configured LLM endpoint")
    key = sub.add_parser("key", help="Manage your own LLM API key (stored on this computer only)")
    key.add_argument("action", choices=("set", "status", "clear"))
    key.add_argument("--provider", choices=keystore.PROVIDERS, default="gemini")
    key.add_argument("--base-url", default="", help="With --provider custom, e.g. http://localhost:11434/v1")
    key.add_argument("--model", default="", help="Model name (see `models`)")
    key.add_argument("--stdin", action="store_true", help="Read the key from standard input instead of prompting")
    chap = sub.add_parser("chapters", help="List chapter numbers")
    chap.add_argument("title")

    for name, help_text in (("ask", "Ask a question about what you've read"), ("recap", "Recap where you are")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("title")
        where = command.add_mutually_exclusive_group(required=True)
        where.add_argument("--chapter", type=int, help="I have finished this chapter")
        where.add_argument("--phrase", help="A line you remember reading last")
        if name == "ask":
            command.add_argument("question")
            command.add_argument("--agent", action="store_true", help="Let the model call search tools (needs an LLM key, see README)")
            command.add_argument("--verbose", action="store_true", help="With --agent: print each tool call")

    args = parser.parse_args(argv)
    if args.command == "key":
        return _key(args)
    semantic = not args.no_embeddings
    library = Library(args.db, embedder=default_embedder() if semantic else None,
                      reranker=default_reranker() if semantic and args.rerank else None)
    try:
        if args.command == "ingest":
            title = args.title or args.file.stem
            chapters = read_book(args.file)
            if not chapters:
                raise ValueError("No readable chapters found")
            print(f"Ingested '{title}': {len(chapters)} chapters, {library.add_work(title, chapters)} passages.")
            if args.embed:
                _embed(library, title)
        elif args.command == "embed":
            _embed(library, args.title)
        elif args.command == "list":
            print("\n".join(library.works()) or "No books yet.")
        elif args.command == "models":
            from .llm import list_models

            for name in list_models():
                print(name)
        elif args.command == "chapters":
            for number, name in library.chapters(args.title):
                print(f"{number:>3}  {name}")
        else:
            if args.phrase:
                horizon, found = library.locate(args.title, args.phrase)
                print(f"Located you at chapter {found.chapter} ({found.chapter_title}):\n  \"{found.text[:160]}...\"\n")
            else:
                horizon = library.horizon_after_chapter(args.title, args.chapter)
            if args.command == "ask" and args.agent:
                session = Session(library, args.title, horizon)
                print(run_agent(session, args.question))
                if args.verbose:
                    print("\nTool calls:\n" + "\n".join(f"  {entry}" for entry in session.log), file=sys.stderr)
            else:
                text, _ = (ask(library, args.title, args.question, horizon) if args.command == "ask"
                           else recap(library, args.title, horizon))
                print(text)
        return 0
    except (KeyError, ValueError, LookupError, OSError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        library.close()


def _embed(library: Library, title: str) -> None:
    if library.embedder is None:
        raise RuntimeError("Semantic search needs: pip install fastembed")
    count = library.embed_work(
        title, progress=lambda done, total: print(f"\r  embedding {done}/{total}", end="", file=sys.stderr))
    print(f"\nEmbedded {count} passages with {library.embedder.name}.")


def _key(args) -> int:
    """`key set` asks for the key without echoing it; nothing is written inside the repository."""
    try:
        if args.action == "set":
            if args.stdin:
                secret = sys.stdin.readline()
            else:
                import getpass

                secret = getpass.getpass(f"Paste your {args.provider} API key (hidden, stored only on this computer): ")
            where = keystore.save(args.provider, secret, args.base_url, args.model)
            print(f"Saved your {args.provider} key in {where}.")
        elif args.action == "clear":
            print("Stored key removed." if keystore.clear() else "No stored key.")
        else:
            stored = keystore.load()
            print(f"stored: provider={stored.get('provider', '(none)')} key={keystore.mask(stored.get('api_key', ''))} "
                  f"model={stored.get('model') or '(default)'} file={keystore.config_path()}")
            print("environment overrides: " + (", ".join(n for n in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
                                                                       "COMPANION_LLM_API_KEY") if os.environ.get(n)) or "none"))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
