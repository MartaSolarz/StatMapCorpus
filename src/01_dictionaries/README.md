# Stage 1 - Keyword dictionaries and the dictionary filter

Weighted phrase dictionaries covering 21 languages, applied with Aho-Corasick to the
alt-text that MapPool inherited from CommonPool/DataComp. Every record is scored as
`score = Σ(positive weights) − Σ(negative weights)`, each phrase counting at most once;
records below −1.0 are dropped, leaving about 54.9 million of 75.9 million records for
Stage 2.

| File | Role |
|---|---|
| `create_dictionary.ipynb` | Builds the dictionaries: language distribution of the sample, choice of languages (at least 1% of the sample), English phrase lists with weights, translation into the other languages with Google Translate (`googletrans`), export to JSON. |
| `keywords/positive_keywords.json`, `keywords/negative_keywords.json` | The frozen dictionaries used by the filter; byte-identical to the two files in the deposit. |
| `keywords/manual_changes/` | The record of phrases added and removed by hand after checking the dictionaries on samples. |
| `apply_dictionary.py` | The filter: downloads the MapPool shards from the Hugging Face Hub and scores every record with Aho-Corasick; writes one result file per shard. |
| `filter_scores.py` | The Stage 1 cut: keeps records with score ≥ −1.0 and merges them into one file per part, the input of Stage 2. `--count-only` reports 54,910,158 of 75,893,474 rows (see the note below). |
| `mappool_shards.json` | The list of the 23,874 MapPool shards, all of which were processed (part -> shard names). |

The dictionaries are frozen: changing them would break the correspondence with the
published `keyword_score` column.

The notebook was run once, in January 2026, on a 10,000-record sample of MapPool that is
not distributed (it carries MapPool alt-text); its outputs are from that run, and it also
needs `langdetect` and `googletrans`.

**Note on counts.** One shard (`part_6/3e031defe480c616a1ed6dc79ed0f6ab`) was written
twice to the Stage 1 output after a failed download was retried, so its 3,067 records
appear twice: the 75,893,474 rows are the 75,890,407 records of MapPool, and the
54,910,158 rows retained are 54,907,936 distinct records. Of these duplicates, 65 reached
a Stage 2 probability of at least 0.5; counts from Stage 2 on (1,675,572) are of distinct
records.

**Inputs not distributed here:** the MapPool shards, which `apply_dictionary.py` downloads
from the Hugging Face Hub (`sraimund/MapPool`).
