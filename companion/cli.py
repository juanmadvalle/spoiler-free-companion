"""Command line: ingest a book, say where you are, ask or recap."""

import argparse
import os
import sys
from pathlib import Path

from . import keystore
from .agent import Session, run_agent
from .answer import ask, recap
from .embed import default_embedder, default_reranker
from .ingest import read_book, read_subtitles
from .subtitles import parse_clock
from .store import Library

DEFAULT_DB = Path.home() / ".spoiler-free-companion.db"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="companion", description="Context for what you've read, without spoilers.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help=f"Library file (default {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("ingest", help="Add an EPUB or .txt you own, or subtitle files (.srt/.vtt), one per episode")
    add.add_argument("file", type=Path, nargs="+", help="A book, or one or more subtitle files in episode order")
    add.add_argument("--keep-order", action="store_true", help="Subtitle files: use the order given, not natural sorting")
    add.add_argument("--title", help="Defaults to the file name")
    add.add_argument("--embed", action="store_true", help="Also build semantic vectors (needs: pip install fastembed)")
    add.add_argument("--lang", choices=("auto", "en", "es"), default="auto", help="Book language (default: detect)")
    embed = sub.add_parser("embed", help="Build semantic vectors for a book already ingested")
    embed.add_argument("title")
    parser.add_argument("--no-embeddings", action="store_true", help="Keyword search only, even if vectors exist")
    parser.add_argument("--rerank", action="store_true", help="Re-order hybrid candidates with a cross-encoder (slower; no measured gain yet)")

    sub.add_parser("list", help="List books")
    sub.add_parser("models", help="List models offered by the configured LLM endpoint")
    serve = sub.add_parser("serve", help="Start the local web interface (127.0.0.1 only)")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    key = sub.add_parser("key", help="Manage your own LLM API key (stored on this computer only)")
    key.add_argument("action", choices=("set", "status", "clear"))
    key.add_argument("--provider", choices=keystore.PROVIDERS, default="gemini")
    key.add_argument("--base-url", default="", help="With --provider custom, e.g. http://localhost:11434/v1")
    key.add_argument("--model", default="", help="Model name (see `models`)")
    key.add_argument("--stdin", action="store_true", help="Read the key from standard input instead of prompting")
    chap = sub.add_parser("chapters", help="List chapter numbers")
    chap.add_argument("title")

    for name, help_text in (("ask", "Ask a question about what you've read"), ("recap", "Recap where you are"),
                            ("characters", "Names met so far"), ("card", "What is known about one name so far")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("title")
        if name == "card":
            command.add_argument("name")
        command.add_argument("--chapter", type=int, help="I have finished this chapter (or episode)")
        command.add_argument("--phrase", help="A line you remember reading or hearing last")
        command.add_argument("--episode", type=int, help="Series/film: the episode you are in (with --at)")
        command.add_argument("--at", help="Series/film: the minute you are at, e.g. 12:30 or 1:02:03 (with --episode)")
        if name == "ask":
            command.add_argument("question")
            command.add_argument("--agent", action="store_true", help="Let the model call search tools (needs an LLM key, see README)")
            command.add_argument("--verbose", action="store_true", help="With --agent: print each tool call")

    args = parser.parse_args(argv)
    if args.command == "key":
        return _key(args)
    if args.command == "serve":
        from .web import serve

        serve(args.db, args.port, open_browser=not args.no_open)
        return 0
    semantic = not args.no_embeddings
    library = Library(args.db, embedder=default_embedder() if semantic else None,
                      reranker=default_reranker() if semantic and args.rerank else None)
    try:
        if args.command == "ingest":
            title, chapters = _read_input(args)
            if not chapters:
                raise ValueError("No readable chapters found")
            passages = library.add_work(title, chapters, None if args.lang == "auto" else args.lang)
            unit = "episodes" if library.is_timed(title) else "chapters"
            print(f"Ingested '{title}' ({library.lang_of(title)}): {len(chapters)} {unit}, {passages} passages.")
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
            horizon = _horizon(library, args)
            if args.command == "characters":
                for entry in library.characters(args.title, horizon):
                    aka = f" (also {', '.join(entry['aliases'])})" if entry["aliases"] else ""
                    print(f"{entry['name']}{aka}: {entry['kind']}, {entry['mentions']} mentions")
            elif args.command == "card":
                print(format_card(library.card(args.title, args.name, horizon)))
            elif args.command == "ask" and args.agent:
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


def _read_input(args) -> tuple[str, list]:
    """(title, chapters) from a book file, or from subtitle files taken as episodes in order."""
    from .subtitles import natural_key

    files = args.file
    if len(files) == 1 and files[0].suffix.lower() not in (".srt", ".vtt"):
        return args.title or files[0].stem, read_book(files[0])
    if any(f.suffix.lower() not in (".srt", ".vtt") for f in files):
        raise ValueError("Give one book file, or only subtitle files (.srt/.vtt)")
    if not args.keep_order:
        files = sorted(files, key=lambda f: natural_key(f.name))
    chapters = [read_subtitles(f, number=i) for i, f in enumerate(files, 1)]
    for number, (file, chapter) in enumerate(zip(files, chapters), 1):
        print(f"  episode {number}: {chapter.title}  ({file.name})", file=sys.stderr)
    default = files[0].stem if len(files) == 1 else (files[0].resolve().parent.name or "Series")
    return args.title or default, chapters


def _horizon(library: Library, args) -> int:
    """The reader's position from --chapter, --phrase, or --episode with --at."""
    if sum([args.chapter is not None, bool(args.phrase), args.at is not None]) != 1:
        raise ValueError('Say where you are with exactly one of: --chapter N, --phrase "...", --episode N --at MM:SS')
    if args.phrase:
        horizon, found = library.locate(args.title, args.phrase)
        print(f"Located you at chapter {found.chapter} ({found.chapter_title}):\n  \"{found.text[:160]}...\"\n")
        return horizon
    if args.at is not None:
        if args.episode is None:
            raise ValueError("--at needs --episode")
        return library.horizon_at_time(args.title, args.episode, parse_clock(args.at))
    if args.chapter == 0:  # 'I have not started': nothing is read yet, as in the web page
        return 0
    return library.horizon_after_chapter(args.title, args.chapter)


def format_card(card: dict) -> str:
    def span(chapters):
        return f"{chapters[0]}-{chapters[-1]}" if len(chapters) > 1 else str(chapters[0])

    aka = f" (also {', '.join(card['aliases'])})" if card["aliases"] else ""
    lines = [f"{card['name']}{aka}: {card['kind']}, {card['mentions']} mentions in chapters {span(card['chapters'])}",
             f"  First seen [{card['first']['ordinal']}] chapter {card['first']['chapter']}: {card['first']['text']}"]
    lines += [f"  Described [{d['ordinal']}]: {d['text']}" for d in card["described_as"]]
    if card["related"]:
        lines.append("  With: " + ", ".join(f"{r['name']} ({r['together']})" for r in card["related"]))
    if card["last"]["ordinal"] != card["first"]["ordinal"]:
        lines.append(f"  Last seen [{card['last']['ordinal']}] chapter {card['last']['chapter']}: {card['last']['text']}")
    return "\n".join(lines)


def _embed(library: Library, title: str) -> None:
    if library.embedder is None:
        raise RuntimeError("Semantic search needs: pip install fastembed")
    count = library.embed_work(
        title, progress=lambda done, total: print(f"\r  embedding {done}/{total}", end="", file=sys.stderr))
    model = library._embedder_for(library._work(title)).name
    print(f"\nEmbedded {count} passages with {model}.")


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
