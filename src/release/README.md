# Release - from the working database to the deposited files

These scripts turned the working SQLite database (`STATMAP_DB`, not published) into the
files of the deposit at https://doi.org/10.58132/FSJGSP. They read data that is not
distributed and are published as the record of how each file was made. The database is
opened read-only by the export scripts.

## The process, step by step

| Step | Script | Writes |
|---|---|---|
| 1. Copy the CLIP embeddings of the corpus maps from the Stage 1 output | `extract_embeddings.py` | table `embeddings` in the database |
| 2. Copy the matched dictionary phrases of the corpus maps from the Stage 1 output | `extract_keywords.py` | columns `pos_keys`, `neg_keys` |
| 3. Compute the checksums of the annotated files and group byte-identical copies | `compute_hashes.py` | columns `sha256`, `phash`, `duplicate_group_id` |
| 4. Export the main table and the embeddings | `export_release.py` | `statmapcorpus_dataset.parquet`, `embeddings_clip_vitl14.npz`, `schema.json` |
| 5. Export every Stage 3 candidate with the outcome of each step | `export_candidates.py` | `candidates.parquet` (stops unless the counts match the data descriptor) |
| 6. Export the validation record | `../../validation/make_validation_csv.py` | `validation.csv` |
| 7. Generate the codebook | `gen_docs.py` | `CODEBOOK.md` (checks every number it states against the files) |
| 8. Add the dictionaries, `tools/download_images.py` and the hand-written README, NOTICE, DATASHEET and VALIDATION | - | - |
| 9. Record sizes and checksums of all files, then check the package | `gen_schema.py`, `verify_package.py` | `schema.json` |

The flag `is_duplicate_byte` exported in step 4 combines `duplicate_group_id` with
`is_duplicated`, written by `src/03_04_retrieval_and_cascade/dedup_flag.py`.
Steps 1 and 2 read `MAPPOOL_DATA`, the Stage 1 output (see `.env.example`); step 3 reads the
downloaded images.

The main table of version 2 is identical in columns and values to that of version 1; only
the file name and the `dataset` entry of its metadata changed. The embeddings file is
carried over from version 1 unchanged: step 4 writes the same arrays, but the file of
version 1 is copied into the package so that its bytes and checksum stay the same.
