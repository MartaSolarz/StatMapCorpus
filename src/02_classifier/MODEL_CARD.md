# Model card - statistical-map classifier

`models/model_final.pkl` is the Stage 2 classifier. Its scores decided which records
went on to Stage 3, where the candidates for StatMapCorpus were drawn. It is a joblib
file holding a dictionary with the fitted `scaler` and `model` (scikit-learn, see
`requirements.txt`), as read by `3_inference/apply_model_large_scale.py`.

## Model

A support vector machine with an RBF kernel and balanced class weights, on standardised
CLIP ViT-L/14 image embeddings (768 dimensions) inherited from MapPool. It outputs the
probability that an image is a statistical map. It scored best among logistic regression,
random forest, XGBoost and SVM models trained on the initial labels
(`evaluation/baseline_results.json`).

## Data

Labels were split once (`random_seed = 42`) into a training set and a fixed test set;
the test identifiers are in `evaluation/test_uids.txt`, the split sizes in
`evaluation/data_preparation_stats.json`.

| Split | Size | YES | NO |
|---|---|---|---|
| Training, initial labels | 3,201 | 591 | 2,610 |
| Training, after active learning | 3,801 | 875 | 2,926 |
| Test (fixed) | 801 | 148 | 653 |

The active-learning batch was selected by entropy from the unlabelled part of the sample
(`evaluation/iteration_1_selection.json`).

## Performance on the test set

| Training set | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Initial labels | 0.823 | 0.721 | 0.959 | 0.977 | 0.863 |
| **After active learning (published)** | **0.850** | **0.763** | **0.959** | **0.980** | **0.882** |

At the 0.5 threshold the published model gives TP 142, FP 44, TN 609, FN 6. Full
metrics: `evaluation/baseline_results.json`, `evaluation/iteration_1_results.json`.

## Use

Of the 54,907,936 distinct records scored (54,910,158 rows; see
`../01_dictionaries/README.md`), 1,675,572 (3.1%) had a probability of at least 0.5 and
were passed on from Stage 2. Stage 3 drew its candidates from those at 0.80 and above, for
budget reasons; the corpus is therefore not a recall-complete sample of the statistical
maps in MapPool.

## Labelling criteria

The rules applied when labelling records YES (statistical map) or NO. They define what
the classifier was trained to pass on; the vision-language gates of Stage 4 apply their
own, stricter definitions (`../03_04_retrieval_and_cascade/multiturn/prompts/`).

1. Do the data refer to spatial units (countries, regions, districts, a grid, isolines)?
   If not, it is probably not a statistical map.
2. Are the data countable or drawn from statistics (population, GDP, votes, cultivated
   area)? Soil or rock type is not statistical.
3. Do they represent mass phenomena, covering many units or counts of people or objects?
4. Are the data quantitative (counts, indices, densities), or qualitative but derived from
   statistics (dominant party, educational structure)? Qualitative data not derived from
   statistics, such as soil type, are not statistical.
5. Is the point of the map to show numerical patterns in space, rather than location or
   routes?

Included: quantitative, ordinal and statistical-qualitative choropleths; isoline maps of
statistical data (population density); maps of statistical data on geometric
units; diagram maps; cartograms.

Decision rules: the map occupies at least 50% of the image; a map with several methods is
YES if at least one meets the criteria; a map whose subject cannot be determined is NO;
strong uncertainty means NO; isolines over seas, air or cloud cover are rejected.

## Limitations

- All labels come from a single annotator.
- The test set is drawn from the same keyword-filtered sample, so the metrics describe
  performance within that population.
- Two records of the initial labels are in both the training and the test split (both
  YES), because the embeddings file held them twice.
