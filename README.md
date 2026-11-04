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

Python 3.10+, standard library only. The LLM is optional: without a key it prints the relevant passages instead of a written answer. See *Choosing an LLM* below to get written answers, including a free option.

```sh
python -m companion ingest mybook.epub --title "My Book"      # .epub or .txt you own
python -m companion chapters "My Book"
python -m companion ask "My Book" --chapter 5 "Who is Marlowe?"        # finished chapter 5
python -m companion ask "My Book" --phrase "a line I just read" "What does she want?"
python -m companion recap "My Book" --phrase "a line I just read"
python -m unittest discover -s tests -t .
```

### Choosing an LLM

Every user brings their own key; nothing is bundled and nothing is stored in the repository. The easiest way:

```sh
python -m companion key set                      # asks for the key without echoing it (Gemini by default)
python -m companion key set --provider anthropic
python -m companion key set --provider custom --base-url http://localhost:11434/v1 --model llama3.2
python -m companion key status                   # shows a masked key and where it is stored
python -m companion key clear
```

The key is saved on your computer only: in the system credential store if the optional `keyring` package is installed (`pip install keyring`), otherwise in `~/.spoiler-free-companion/config.json` with owner-only permissions, which is plain text. Environment variables always win over the stored key:

| Provider | Variables | Cost |
|---|---|---|
| Google Gemini (free tier) | `GEMINI_API_KEY` from [aistudio.google.com](https://aistudio.google.com) | free, rate-limited; on the free tier Google may use submitted text to improve its products, so passages of the book you ask about are sent to Google |
| Local Ollama | `COMPANION_LLM_BASE_URL=http://localhost:11434/v1`, `COMPANION_LLM_MODEL=<model>` | free, private, slower |
| Any OpenAI-compatible endpoint | `COMPANION_LLM_BASE_URL`, `COMPANION_LLM_API_KEY`, `COMPANION_LLM_MODEL` | varies |
| Anthropic | `ANTHROPIC_API_KEY` (and `pip install anthropic`), `COMPANION_MODEL` | paid |

`python -m companion models` lists the model names the configured endpoint offers; pick one with `COMPANION_LLM_MODEL` (the default is `gemini-2.5-flash`, which may have been retired). Anthropic is used if its key is set, otherwise the compatible endpoint. Only the retrieved passages, never the whole book, are sent. Small free or local models can be unreliable at tool calling, so `--agent` may need a stronger model than the plain `ask`.

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

Measured on the cached real books (`scripts/eval_semantic.py`, 31 hand-written paraphrase questions, a relevant passage in the top 5): keyword-only 14/31, hybrid 18/31. Queries get the instruction prefix BGE models expect (worth about one question). Exact-word lookups are essentially unchanged (237 to 236 of 238 in the top 3) and lookups with changed word forms dip (233 to 223 of 238). The question set is small and the model compact, so treat these as indicative; many paraphrases still miss.

`--rerank` adds an optional cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, about 80 MB) that re-orders the top 40 hybrid candidates. In a first comparison it looked like +3 questions, but in the final integrated run it fixed 3 and broke 3 (18/31 either way), so it stays off by default. Larger rerankers (MiniLM-L12, Jina) did worse on this set.

**Other embedding models.** Set `COMPANION_EMBED_MODEL` and run `embed` again (vectors are stored per model, so several can coexist). `python scripts/compare_models.py` compared three on the Alice and Frankenstein questions (17, hit in the top 5): `BAAI/bge-small-en-v1.5` (default, 67 MB, about 5 passages/s) 9, `BAAI/bge-base-en-v1.5` (210 MB, about 2/s) 11, `snowflake/snowflake-arctic-embed-m` (430 MB, about 1.6/s) 10. A gain of two questions out of 17 is within noise for this small set, so the default stays small; try `bge-base` if indexing time and disk do not matter to you.

**Passage size.** Passages are at most 700 characters, cut at sentence ends and never across a chapter. `python scripts/compare_chunking.py` tried other limits on the Alice and Frankenstein questions (17; keyword / hybrid hits in the top 5): 250 chars 7 / 7, 350 chars 5 / 8, 500 chars 4 / 9, 700 chars 9 / 11, 1000 chars 9 / 11, 1400 chars 8 / 12. Shorter passages were clearly worse and longer ones no better, so 700 stays (changing the limit means re-ingesting books). Cutting oversize paragraphs at sentence ends also made `--phrase` locating exact on all 336 sampled phrases.

### Aliases and future questions

- **Aliases:** at ingest, multi-word proper names ("Sherlock Holmes") are detected with the position of their first appearance. A search for "Sherlock" also looks for "Holmes" only when, at your position, that name is the single unambiguous match for both words. A name first seen later (say a second "Mark Reed") can never influence an earlier answer, and an ambiguous surname such as "Bennet" is left alone. Nicknames ("Lizzy") are not detected.
- **Future questions:** "How does it end?", "Will she die?", "¿Cómo acaba?" and similar get an immediate "not established up to this point" with no search and no model call. The patterns are deliberately narrow, so a question about the past that is misread can simply be rephrased; the horizon filter still protects everything else.

### Agent mode

`ask --agent` (needs an LLM, see below) lets the model call search tools in a loop: `search_passages`, `recent_passages`, `read_chapter` and `ask_user`. Add `--verbose` to print each tool call.

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
