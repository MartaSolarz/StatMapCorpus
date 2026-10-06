# Technical Validation

The code behind the three validation exercises of the data descriptor: the model labels
against two human coders, the repeatability of the model labels, and a blind audit of the
images the cascade rejected.

## Reproducing the numbers

The five analysis scripts compute every number of those three subsections from two files
of the deposit, the main table `statmapcorpus_dataset.parquet` and the validation
record `validation.csv`:

```bash
pip install -r ../requirements.txt
python coders/agreement.py       --deposit <deposit directory>
python coders/precision.py       --deposit <deposit directory>
python coders/extra_analyses.py  --deposit <deposit directory>
python retest/analyze_retest.py  --deposit <deposit directory>
python audit/analyze_audit.py    --deposit <deposit directory>
```

Each prints its tables and writes a machine-readable copy to `out/` in its directory.

| Script | Technical Validation |
|---|---|
| `coders/agreement.py` | disagreements between the coders and Cohen's kappa per label |
| `coders/precision.py` | weighted precision under the three reference standards A, B and C (Table 6) |
| `coders/extra_analyses.py` | complete type sets, the three criteria jointly, coder independence (McNemar), model confidence, error patterns |
| `retest/analyze_retest.py` | repeatability of Gates 3 and 4, and instability as a signal of error |
| `audit/analyze_audit.py` | share of rejected images that met all criteria, per stratum and extrapolated |

## How the data were collected

These scripts read the working database, the image cache or the Anthropic API, which are
not distributed; they are the record of how the samples and labels came about.

| Exercise | Sample | Labels | Into `validation.csv` |
|---|---|---|---|
| Coders | `coders/sample_validation.py`: 20 maps per type and all 17 maps without lettering, 157 in all; the probabilities of inclusion follow from its sampling record | two coders, following `codebook/` | `make_validation_csv.py` |
| Repeatability | `retest/make_sample.py`: the 157 maps plus 143 drawn at random from the rest of the corpus | `retest/run_retest.py`: two re-runs of Gates 3 and 4 with the production code, prompts and models | `make_validation_csv.py` |
| Audit | `audit/make_sample.py`: 130 images from five rejection strata and 20 corpus maps as controls (`DECOY`) | one blind coding: does the image meet all inclusion criteria? | `make_validation_csv.py` |

`codebook/gates.yaml` holds the questions and definitions the coders saw and
`codebook/coder_instructions.md` the briefing they read; both are translations of the
Polish originals. The interfaces used to screen and code the maps are not included. The
thematic map types are numbered as in the codebook: M1 choropleth, M2 diagram map, M3
isoline map, M4 dot density, M5 heat map, M6 cartogram, M7 flow map.

## Analysis plans

`ANALYSIS_PLANS.md` condenses the analysis plans of the three exercises, each recorded
before its data were collected, and lists the departures from them.
