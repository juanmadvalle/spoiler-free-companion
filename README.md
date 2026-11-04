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

### Search quality

The index stems words ("crying" finds "cried") and re-ranks the top candidates by how many distinct query terms each passage covers. `python scripts/eval_search.py` measures this on the cached real books: queries of three remembered words, all inflected, find a passage containing them in the top 3 for 99.0% of 585 cases (0.5% without stemming), and noisy queries rise from 92.8% to 98.1% with the re-ranking. Libraries built earlier are upgraded in place on first open. Synonyms and paraphrases are not handled.

### Semantic search (optional, local, no API key)

`pip install fastembed` (about 130 MB model, downloaded once to `~/.cache/spoiler-free-companion/models`), then:

```sh
python -m companion ingest mybook.epub --title "My Book" --embed
python -m companion embed "My Book"        # for a book already ingested
```

Questions are then matched by meaning as well as by words ("a child weeping" finds "the girl cried"). Use `--no-embeddings` to force keyword search. Embedding is slow on CPU (about 5 passages per second: a novel takes a few minutes) but happens once.

Each passage is embedded on its own, and only the vectors of passages at or before your horizon are ever loaded from the database, so the spoiler guarantee is unchanged; tests check this on all four real books. Passages that contain nearly all the query words stay first (exact lookups), the rest are merged with the semantic ranking.

Measured on the cached real books (`scripts/eval_semantic.py`, 17 hand-written paraphrase questions, hit in the top 5): keyword-only 5/17, hybrid 8/17. Exact-word lookups are unchanged (237/238 in the top 3) and inflected lookups dip slightly (233 to 222 of 238). The question set is small and the model is a compact one, so treat these as indicative; many paraphrases still miss.

### Aliases and future questions

- **Aliases:** at ingest, multi-word proper names ("Sherlock Holmes") are detected with the position of their first appearance. A search for "Sherlock" also looks for "Holmes" only when, at your position, that name is the single unambiguous match for both words. A name first seen later (say a second "Mark Reed") can never influence an earlier answer, and an ambiguous surname such as "Bennet" is left alone. Nicknames ("Lizzy") are not detected.
- **Future questions:** "How does it end?", "Will she die?", "¿Cómo acaba?" and similar get an immediate "not established up to this point" with no search and no model call. The patterns are deliberately narrow, so a question about the past that is misread can simply be rephrased; the horizon filter still protects everything else.

### Agent mode

`ask --agent` (needs `ANTHROPIC_API_KEY` and `pip install anthropic`) lets the model call search tools in a loop: `search_passages`, `recent_passages`, `read_chapter` and `ask_user`. Add `--verbose` to print each tool call.

The horizon stays under the harness's control: no tool takes a horizon, book or position argument, extra arguments the model invents are ignored, and every result is capped at the horizon in SQL and re-checked. Tests drive the loop with a scripted fake model that tries to read ahead. Not yet exercised against the live API.

`--phrase` finds your place from a line you remember and prints the match so you can confirm it's the right spot. If a phrase is repeated, pick a more distinctive one.

## Copyright

Books are processed locally from files you provide; nothing is uploaded except the few retrieved passages sent to the LLM when a key is set. The library file stores the book text, so keep it private and don't share it. The `demo/` book (Project Gutenberg, public domain) is downloaded, not bundled.

## Limits

- Reading position is per chapter or per remembered phrase, not page numbers (they vary by edition).
- Retrieval is keyword-based (SQLite FTS5 with stemming), so questions that share no words with the text, such as pure paraphrases or synonyms, may still miss. Embeddings are the natural next step; keep the same SQL horizon filter.
- Chapter titles are visible in `chapters`; so are headings in the text itself.
- Books only. Video would need subtitle files (`.srt`) with timestamps as positions.
- Questions that ask about the future ("how does it end?") get "not established up to this point" by design.

## License

MIT, see [LICENSE](LICENSE).
