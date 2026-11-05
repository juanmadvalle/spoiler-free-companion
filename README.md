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

### Films and series (subtitles)

Bring your own subtitle files (`.srt` or `.vtt`); none are bundled, and they are processed on your computer. One file is one episode (or the whole film):

```sh
python -m companion ingest "Show.S01E01.srt" "Show.S01E02.srt" "Show.S01E03.srt" --title "My Show"   # episodes in natural order
python -m companion ask "My Show" "Who is the man in the grey coat?" --episode 2 --at 31:30            # I am 31:30 into episode 2
python -m companion ask "My Show" "What happened so far?" --chapter 1                                  # I have finished episode 1
python -m companion characters "My Show" --episode 2 --at 31:30
```

In the web page, choose several subtitle files at once; a book list entry then shows episodes, and the position box gains "I am in episode N, at minute M". Answers and cards cite the episode and the minute (`S01E02 · 1:00`). Everything else (search, character cards, the agent, Spanish or English) works as for books.

How the minute is applied: subtitle lines are grouped into short passages (at most about 400 characters, 60 seconds, split at silences of 12 seconds or more), and you have "seen" a passage only once it has **fully ended** by your minute. A passage still being spoken at that minute is not included, so you never get a line you have not heard. Locating by a remembered line likewise stops one passage before the one that contains it. This costs a few seconds of context and guarantees the rule. Hearing-impaired captions such as `[door slams]` stay in the text; promotional lines ("Subtitles by …") are dropped; Windows/Latin-1 encodings are handled. Season/episode labels come from the file name (`S01E02`, `2x03`), otherwise from the cleaned name.

Tested with synthetic series and a large check that turns *Sherlock Holmes* into 12 episodes of generated subtitles: at 8 episode-and-minute positions no search, recap, character list or card contains a word that is only said later, and deliberately counting passages that merely *start* before the minute makes the same check fail. Not tested: real subtitle files (none can be bundled), word-level timing inside a passage, and video files themselves; you give the position by hand (a media player's current time), the tool does not watch the video.

### Character cards

`python -m companion characters "My Book" --chapter 5` lists the names met so far and `card "My Book" Holmes --chapter 5` shows what is known about one: other names it goes by (Sherlock Holmes → Holmes), first and last appearance with passage numbers, sentences that say who they are ("X was a ...", "X, the ..."), and who they appear with. In the web page it is the "Characters so far" section; clicking a name opens its card, and one more click asks the AI who they are. The agent has a `character_card(name)` tool.

No AI is needed. Cards are computed on the fly from an index of where each name occurs, always with `ordinal <= your position`, and even *whether a word counts as a name* is decided from the mentions up to your position. So a character who only appears later has no entry and no card yet, and a name never shows who they end up appearing with. Tests check this on all six real books: for ten positions each, every name, alias, excerpt and "appears with" entry in the cards must use only words that exist at or before that position. Two real bugs came out of those tests and were fixed (contractions like "I'll" counted as names; an excerpt cut in the middle of a word).

Limits: it only knows capitalised names that appear mid-sentence, so it cannot tell a person from a place without cues (speech verbs, honorifics); those are labelled `character` or `other`. Descriptions are the book's own sentences, not a summary, and "appears with" counts shared passages, not relationships. Nicknames and pronouns are not resolved.

### Spanish books

Spanish is supported alongside English. The language of each book is detected when you add it (or set it with `ingest --lang es`, or the language selector on the web page) and shown as `ES`/`EN`. Spanish books get:

- a Spanish Snowball stemmer written for this project (`companion/lang.py`), checked to give identical results to the reference implementation on all 24,535 distinct words of Don Quijote and Lazarillo; passages are indexed under both their accented and unaccented spelling, so questions typed without accents still match;
- Spanish chapter detection (`Capítulo`, `Tratado`, `Parte`, `Carta 1`…), where an unnumbered `Prólogo`, `Prefacio`, `Introducción`, `Índice`, notes and a part's title page are dropped as paratext because they often reveal the plot (a letter inside a chapter, such as "Carta de don Quijote a Dulcinea", is not mistaken for a chapter);
- names with accents and titles (`don`, `doña`, `señor`…), Spanish dialogue dashes and `¿ ¡`, and Spanish refusals ("Eso depende de lo que viene después…") chosen by the language of your question;
- a multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`, about 220 MB, `COMPANION_EMBED_MODEL_MULTILINGUAL` to change it) when you build semantic vectors. The cross-encoder reranker stays English-only and is skipped for Spanish books.

Tested on Lazarillo de Tormes (7 tratados) and Don Quijote (all 126 chapters) from Project Gutenberg with the same anti-spoiler tests as the English books, plus `--phrase` locating (731 of 732 sampled phrases exact, none placed too late). `python scripts/eval_spanish.py` measures what Spanish support adds, in lookups of three remembered words (top 3):

| | Lazarillo | Don Quijote |
|---|---|---|
| exact spelling, before / now | 100% / 100% | 100% / 100% |
| another form of the same word, before / now | 86% / 99% | 75% / 100% |
| typed without accents, before / now | 100% / 100% | 100% / 100% |

Read with care: the "other form" variants are real forms found elsewhere in the same book, but "same word" is decided by the same stemmer, so this shows the pipeline does what it claims rather than independent proof of quality. On 11 hand-written paraphrase questions Spanish keyword search and the old English path tied (10 of 11), and on the 5 Lazarillo questions the multilingual embeddings scored 4 against 5 for keywords only, so on this tiny set they have not yet shown a benefit. Other languages are not supported yet; they would need a stemmer and heading rules in the same places.

### Live check of agent mode

`python scripts/live_agent_check.py` runs the agent against *your* configured LLM on a synthetic book whose last chapter hides a secret, with the reader at chapter 1: a direct question naming the secret, an explicit "use read_chapter on chapter 3", and an instruction planted in chapter 1 telling the model to ignore its limits. It passes only if the secret never reaches the model through a tool and never appears in an answer. Last run (Gemini free tier, `gemini-flash-lite-latest`): 3 of 3 passed; the model did call `read_chapter(3)` and received nothing. A separate live run on Sherlock Holmes at chapter 1 showed the agent searching three times for a character who had not appeared yet and correctly reporting no mention.

### Web interface

```sh
python -m companion serve          # opens your browser; --port N to change the port, --no-open to skip the browser
```

A single page where you add a book (EPUB or text), say how far you have read (a finished chapter, or paste a line you remember and it places you), then ask questions or request a recap. Answers show their citations as buttons that open the exact passages used, and an AI settings dialog lets each user paste their own key (the same storage as `key set`). It works in Spanish or English following the browser language, in light or dark mode, and on a phone-sized screen. Your position is remembered per book in the browser.

Because it handles your key, the server is locked down: it listens on `127.0.0.1` only, the link it prints contains a one-time token that every request must carry, requests with a foreign `Host` header (DNS rebinding) or non-JSON bodies (cross-site form posts) are refused, uploads and JSON bodies have size limits, the page is served with a strict Content-Security-Policy (no external resources, no inline handlers), the uploaded file name is never used as a path, and the key is never sent back to the page (only a masked form). Chapter numbers beyond the end of a book are rejected instead of marking the whole book as read.

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

`python -m companion models` lists the model names the configured endpoint offers; pick one with `COMPANION_LLM_MODEL` (the default is Google's `gemini-flash-lite-latest` alias, which follows the current lite model and was the one that answered when the larger flash models were overloaded on the free tier; fixed names such as `gemini-2.5-flash` get retired). Anthropic is used if its key is set, otherwise the compatible endpoint. Only the retrieved passages, never the whole book, are sent. Small free or local models can be unreliable at tool calling, so `--agent` may need a stronger model than the plain `ask`.

**Tested live** (November 2026, Gemini free tier, `gemini-3.1-flash-lite`, Sherlock Holmes): a plain `ask` answers from the passages with citations, says "not established" for a character who has not appeared yet, and `--agent` ran a tool call and answered with citations. Larger flash models often returned HTTP 503 (overloaded) on the free tier; requests are retried with a short wait on 429/5xx, timeouts and transient network errors. Gemini 3 needs its thought signature echoed back with each tool call, which the adapter does. Not yet verified live: the agent refusing to read ahead (covered by tests with a scripted model).

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

- **Aliases:** at ingest, multi-word proper names ("Sherlock Holmes") are detected and recorded at the passage where they become established, their second appearance, so a link is never available before the passages that justify it (libraries ingested earlier keep first-appearance positions until you add the book again). A search for "Sherlock" also looks for "Holmes" only when, at your position, that name is the single unambiguous match for both words. A name first seen later (say a second "Mark Reed") can never influence an earlier answer, and an ambiguous surname such as "Bennet" is left alone. Nicknames ("Lizzy") are not detected.
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
