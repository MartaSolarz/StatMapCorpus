# StatMapCorpus – code

Code that built and validated **StatMapCorpus**, a corpus of 23,549 English-language
statistical maps from the web annotated by thematic map type.

- **Dataset (version of record):** https://doi.org/10.58132/FSJGSP – Dane Badawcze UW

The data files (main table, embeddings, candidate file, validation record) and the
documentation of every field (CODEBOOK.md) are in the deposit, not here.

## Layout

| Path | Contents | Stage in the data descriptor |
|---|---|---|
| `src/01_dictionaries/` | Weighted multilingual keyword dictionaries and the Aho-Corasick filter over MapPool alt-text | Stage 1 |
| `src/02_classifier/` | SVM-RBF classifier over CLIP ViT-L/14 embeddings, active learning, inference; `models/model_final.pkl` is the published model | Stage 2 |
| `src/03_04_retrieval_and_cascade/` | Deduplication, sampling, URL validation and download; the vision-language cascade with its prompts (`multiturn/prompts/`) | Stages 3 and 4 |
| `src/release/` | Export of the deposited files from the working database | – |
| `validation/` | Technical Validation: coder agreement and precision, test-retest, audit of rejected images; analysis plans | – |
| `tools/download_images.py` | Retrieves the images and verifies them against the published checksums (also in the deposit) | – |

The gates of Stage 4 are named differently in the code:

| Data descriptor | Code | Model |
|---|---|---|
| Gate 1 | size gate (`MIN_IMAGE_DIM_PX = 400`) | none |
| Gate 2 | `step1` | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) |
| Gate 3 | `step2` | Claude Haiku 4.5 |
| Gate 4 | `step3` | Claude Sonnet 4.6 (`claude-sonnet-4-6`) |

## What runs from a clone

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

- **`validation/`** reproduces the numbers of the Technical Validation (coders, retest,
  audit) from two files of the deposit, `statmapcorpus_dataset.parquet` and
  `validation.csv`. See `validation/README.md`.
- **`tools/download_images.py`** fetches the images of the corpus:
  `python tools/download_images.py --data statmapcorpus_dataset.parquet --out images/ --verify-phash`

`src/` is a record of how the corpus was produced rather than a pipeline that reruns end to
end. It reads MapPool shards (obtain them from MapPool), downloaded images and a working
SQLite database that is not published because it holds raw model responses and local
paths. Paths are taken from environment variables listed in `.env.example`. The model
annotations cannot be regenerated identically: model versions will be retired and
vision-language output is not deterministic, which is why the labels are published as
frozen files with checksums.

## Licence and citation

Code: MIT (`LICENSE`). The dataset is licensed separately (CC BY 4.0, see the deposit).
Source images remain with their owners and are not distributed.

Please cite the data descriptor and the dataset; `CITATION.cff` describes this code.
Candidates derive from MapPool: Schnürer, R. (2024). *MapPool – Bubbling up an extremely
large corpus of maps for AI.*

## Contact and funding

Marta Solarz, Faculty of Geography and Regional Studies, University of Warsaw –
m.solarz2@uw.edu.pl · ORCID [0009-0001-4134-8500](https://orcid.org/0009-0001-4134-8500)

Funded by the University of Warsaw Excellence Initiative – Research University (IDUB),
Action IV.4.1, Microgrants for Doctoral Candidates 2026, grant BOB-IDUB-622-272-2026.
