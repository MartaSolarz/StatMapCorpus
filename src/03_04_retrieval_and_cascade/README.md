# Stages 3 and 4 - Retrieval and the vision-language cascade

One program covers both stages: `run_pipeline.py` is a command-line tool whose subcommands
run the steps below, `config.py` holds every threshold, and `db.py` and `multiturn/db.py`
keep the state of each record in one SQLite database (`STATMAP_DB`, not published). Every
step processes only the records still pending, so it can be interrupted and rerun.

## The process, step by step

| Step | Command | Result |
|---|---|---|
| **Stage 3** | | |
| 1. Normalise URLs, keep one record per normalised URL (the highest probability), drop three excluded domains | `python clean_pool.py --input <Stage 2 predictions>` | 1,675,572 → 1,285,562 records (366,771 duplicate URLs, 23,239 from excluded domains), `full_data/predictions_clean.parquet` |
| 2. Keep records with probability ≥ 0.80 and draw at most five per domain | `python run_pipeline.py sample` | 850,724 → 203,371 candidates from 104,557 domains |
| 3. Check the URLs (HEAD requests, 30 in parallel, 8 s timeout) | `python run_pipeline.py validate-urls` | 122,963 alive, 80,277 dead, 131 errors |
| 4. Download the images (15 in parallel, 15 s timeout) and record their size and format | `python run_pipeline.py download` | 116,573 decoded, 5,770 not images, 620 failed |
| **Stage 4** | | |
| 5. Gates 1-4 for each image in turn | `python run_pipeline.py phase1 --all` | see below |
| 6. Flag byte-identical files (same size and dimensions, confirmed by MD5); in each group the copy that went furthest in the pipeline is kept, then the majority verdict of a later quality-annotation step of a separate project, then the highest resolution and the lowest identifier | `python dedup_flag.py` | `is_duplicated`, used by `src/release/` |

`python run_pipeline.py status` prints the counts at any point.

Step 2 draws within each domain at random; the seed of the published run was not recorded,
so the draw itself is documented by the deposited `candidates.parquet`, which lists all
203,371 candidates with the outcome of every step.

## Stage 4 - the gates

Each image is driven through the gates by one worker, so a rejection stops the chain and
costs nothing downstream.

| Gate | Code | Model | Decides | Passed |
|---|---|---|---|---|
| Gate 1 | `size_gate` | none | shorter side at least 400 px (`MIN_IMAGE_DIM_PX`) | 68,035 |
| Gate 2 | `step1` | Claude Haiku 4.5 | is a map; the map covers at least about half of the image; one-sentence description | 67,273 |
| Gate 3 | `step2` | Claude Haiku 4.5 | statistical map; data on administrative or statistical units; quantitative data; language of the lettering (must be `en`) | 23,646 |
| Gate 4 | `step3` | Claude Sonnet 4.6 | the seven thematic map types, legend type, confidence; at least one type required | 23,549 |

The phase-1 chain also runs `step4`, a rule without a model call that flags choropleth-only
maps with a classed legend for a separate project; it does not affect the corpus.

The prompts are in `multiturn/prompts/` (`system.md` shared by all three model calls,
`step1.md`-`step3.md` per gate). They are published exactly as sent, including the
project's working title and the term "cartographic method" for what the data descriptor
calls a thematic map type. Each call sends the gate's text first, marked for prompt
caching, then the image downscaled to at most 1,024 px, so the cached prefix is shared
across images. Responses are parsed by a routine tolerant of code fences and truncated
output. Model identifiers, token limits and the number of workers are set in `config.py`
and `multiturn/config.py`; the annotations were generated on 1-2 August 2026.

## Paths

| Variable or path | What it is |
|---|---|
| `full_data/predictions_clean.parquet` | Output of step 1, input of step 2 |
| `STATMAP_DB` | The working SQLite database (default `pipeline.db` here) |
| `STATMAP_IMAGE_CACHE` | Where the downloaded images are stored |
| `ANTHROPIC_API_KEY` | Needed for Gates 2-4 |

Variables are read from the environment or from `.env` at the root of the repository
(see `.env.example`).
