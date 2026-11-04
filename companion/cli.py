"""Command line: ingest a book, say where you are, ask or recap."""

import argparse
import sys
from pathlib import Path

from .agent import Session, run_agent
from .answer import ask, recap
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

    sub.add_parser("list", help="List books")
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
            command.add_argument("--agent", action="store_true", help="Let the model call search tools (needs ANTHROPIC_API_KEY)")
            command.add_argument("--verbose", action="store_true", help="With --agent: print each tool call")

    args = parser.parse_args(argv)
    library = Library(args.db)
    try:
        if args.command == "ingest":
            title = args.title or args.file.stem
            chapters = read_book(args.file)
            if not chapters:
                raise ValueError("No readable chapters found")
            print(f"Ingested '{title}': {len(chapters)} chapters, {library.add_work(title, chapters)} passages.")
        elif args.command == "list":
            print("\n".join(library.works()) or "No books yet.")
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
