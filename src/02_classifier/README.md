# Stage 2 - Image classifier trained with active learning

A support vector machine with an RBF kernel over the CLIP ViT-L/14 image embeddings that
MapPool carries for every record. It was trained on manually labelled samples, extended by
active learning, and applied to the 54.9 million records left by Stage 1.
Records with a predicted probability of at least 0.5 were passed on; Stage 3 drew its
candidates from those at 0.80 and above.

## The process, step by step

| Step | File | Reads | Writes |
|---|---|---|---|
| 1. Drop the embeddings from the Stage 1 output | `1_sampling/strip_embeddings.py` | Stage 1 output (`MAPPOOL_DATA`) | `data/sample/all_no_embeddings.parquet` |
| 2. Draw a 200,000-record sample, stratified by keyword score | `1_sampling/sampling.ipynb` | the file above | `data/sample/sample_200k.parquet` |
| 3. Attach the embeddings to the sample | `1_sampling/get_sample_embeddings.py` | the sample, Stage 1 output | `data/sample/sample_200k_with_embeddings.parquet` |
| 4. Label 4,000 records from the sample | criteria in `MODEL_CARD.md` | - | `data/baseline/baseline.parquet` |
| 5. Split the labels into training and a fixed test set; the rest is the pool | `2_training/prepare_train_data.py` | steps 3-4 | `data/train.parquet`, `data/test.parquet`, `data/pool.parquet` |
| 6. Compare logistic regression, random forest, XGBoost and an RBF-kernel SVM on the initial labels; the SVM scores best | `2_training/baseline_model.py` | step 5 | `models/`, `results/` (→ `evaluation/baseline_results.json`) |
| 7. Select the most uncertain pool records for labelling | `2_training/active_learning.py` | model, pool | `iterations/iteration_N/` (→ `evaluation/iteration_N_selection.json`) |
| 8. Label them and refit | `2_training/retrain_model.py --iteration N` | step 5, `iterations/iteration_N/annotated.parquet` | `models/model_iteration_N_*.pkl`, `results/` (→ `evaluation/iteration_N_results.json`); the iteration-1 model is published as `models/model_final.pkl` |
| 9. Score all Stage 1 records | `3_inference/apply_model_large_scale.py` | `models/model_final.pkl`, Stage 1 output | the 1,675,572 distinct records with probability ≥ 0.5 (3.1% of the 54,907,936 distinct records scored), the input of Stage 3 |

Paths are relative to this directory; `data/`, `iterations/`, `models/` and `results/` are
created by the scripts. `MAPPOOL_DATA` is the directory of `part_*.parquet` files written by
`src/01_dictionaries/filter_scores.py`.

## What is published here

| Path | Contents |
|---|---|
| `models/model_final.pkl` | The classifier whose scores decided which records went on to Stage 3. Its training data, metrics and labelling criteria are in `MODEL_CARD.md`. |
| `evaluation/` | Metrics of the initial and the published model as written by the training scripts, the selection statistics of the active-learning batch, the split sizes, and the 801 test identifiers. |

The labels were entered in a small local annotation tool, not included; for the initial
4,000 it drew records from the sample at random within score bins (unseeded), and for active
learning it showed the records selected in step 7. The intermediate data
and the labels (`data/`, `iterations/`) are not published.
