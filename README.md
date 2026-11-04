# Spoiler-free reading companion

Tell it which book you're reading and where you are; ask questions or get a recap using **only what you've already read**.

## How spoilers are prevented

The guarantee is structural, not a prompt instruction:

1. Every passage is stored with its reading-order position (`ordinal`, `chapter`).
2. Your position becomes a **horizon**. Every query filters `ordinal <= horizon` **in SQL**, so later text never leaves the database.
3. The model only receives those passages and must cite them. If they don't answer the question, it says so.
4. `check_horizon` fails loudly if a passage past the horizon is ever retrieved, and the tests plant secret words in later chapters and try hostile queries to prove none leak.

`tests/test_real_books.py` repeats this on four real public-domain books (Sherlock Holmes, Pride and Prejudice, Alice, Frankenstein; downloaded from Project Gutenberg and cached in `tests/data/`, skipped offline). For many reading positions it takes the words that appear only *after* that point, queries with them through every read path (search, recap, chapter reads, the agent tools, even with a forged `horizon` argument) and fails if any appears. It also checks that characters are invisible until their first chapter and that `--phrase` locates the right chapter (318 of 319 sampled phrases, none placed too late). Deliberately breaking the filter makes these tests fail.

Plain-text ingestion drops everything before the first chapter heading and prefaces, introductions and notes, since they often reveal the plot.

## Usage

Python 3.10+, standard library only. The LLM is optional: without `ANTHROPIC_API_KEY` it prints the relevant passages instead of a written answer. With a key (and `pip install anthropic`) it answers in prose; set `COMPANION_MODEL` to change the model.

```sh
python -m companion ingest mybook.epub --title "My Book"      # .epub or .txt you own
python -m companion chapters "My Book"
python -m companion ask "My Book" --chapter 5 "Who is Marlowe?"        # finished chapter 5
python -m companion ask "My Book" --phrase "a line I just read" "What does she want?"
python -m companion recap "My Book" --phrase "a line I just read"
python -m unittest discover -s tests -t .
```

### Agent mode

`ask --agent` (needs `ANTHROPIC_API_KEY` and `pip install anthropic`) lets the model call search tools in a loop: `search_passages`, `recent_passages`, `read_chapter` and `ask_user`. Add `--verbose` to print each tool call.

The horizon stays under the harness's control: no tool takes a horizon, book or position argument, extra arguments the model invents are ignored, and every result is capped at the horizon in SQL and re-checked. Tests drive the loop with a scripted fake model that tries to read ahead. Not yet exercised against the live API.

`--phrase` finds your place from a line you remember and prints the match so you can confirm it's the right spot. If a phrase is repeated, pick a more distinctive one.

## Copyright

Books are processed locally from files you provide; nothing is uploaded except the few retrieved passages sent to the LLM when a key is set. The library file stores the book text, so keep it private and don't share it. The `demo/` book (Project Gutenberg, public domain) is downloaded, not bundled.

## Limits

- Reading position is per chapter or per remembered phrase, not page numbers (they vary by edition).
- Retrieval is keyword-based (SQLite FTS5), so questions that share no words with the text may miss. Embeddings are the natural next step; keep the same SQL horizon filter.
- Chapter titles are visible in `chapters`; so are headings in the text itself.
- Books only. Video would need subtitle files (`.srt`) with timestamps as positions.
- Questions that ask about the future ("how does it end?") get "not established up to this point" by design.

## License

MIT, see [LICENSE](LICENSE).
