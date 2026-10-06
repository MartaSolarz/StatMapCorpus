import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PKG = Path(os.environ.get("STATMAP_OUT", str(ROOT / "build" / "packages")))
VER = "v2"
BASE = PKG / f"statmapcorpus-{VER}"
MAIN = BASE / "statmapcorpus_dataset.parquet"
CANDIDATES = BASE / "candidates.parquet"
VALIDATION = BASE / "validation.csv"


def facts(df: pd.DataFrame) -> dict:
    n = len(df)
    pct = lambda k: round(100 * k / n, 1)
    lic = df.legend_is_classed
    choro_only = (df.methods_count == 1) & df.method_choropleth
    dup_phash = df.phash.duplicated(keep=False)
    return {
        "rows": n,
        "columns": len(df.columns),
        "https": int(df.url.str.startswith("https").sum()),
        "domains": df.domain.nunique(),
        "sha256_distinct": df.sha256.nunique(),
        "phash_distinct": df.phash.nunique(),
        "has_text_false": int((df.has_text == False).sum()),
        "choropleth": int(df.method_choropleth.sum()),
        "choropleth_pct": pct(df.method_choropleth.sum()),
        "methods_count": df.methods_count.value_counts().sort_index().to_dict(),
        "legend_null": int(lic.isna().sum()),
        "legend_true": int((lic == True).sum()),
        "legend_false": int((lic == False).sum()),
        "confidence": df.classification_confidence.value_counts().to_dict(),
        "duplicates": int(df.is_duplicate_byte.sum()),
        "grouped_rows": int(df.duplicate_group_id.notna().sum()),
        "groups": df.duplicate_group_id.nunique(),
        "phash_shared_rows": int(dup_phash.sum()),
        "phash_shared_ungrouped": int((dup_phash & df.duplicate_group_id.isna()).sum()),
        "choropleth_only": int(choro_only.sum()),
        "choropleth_only_null": int((choro_only & lic.isna()).sum()),
    }


EXPECTED_MAIN = {
    "rows": 23_549, "columns": 30, "https": 18_284, "domains": 16_374,
    "sha256_distinct": 22_499, "phash_distinct": 20_399, "has_text_false": 17,
    "choropleth": 20_888, "choropleth_pct": 88.7,
    "methods_count": {1: 21_158, 2: 2_278, 3: 110, 4: 2, 5: 1},
    "legend_null": 6_053, "legend_true": 14_857, "legend_false": 2_639,
    "confidence": {"high": 18_644, "medium": 4_899, "low": 6},
    "duplicates": 1_050, "grouped_rows": 1_886, "groups": 836,
    "phash_shared_rows": 5_180, "phash_shared_ungrouped": 3_294,
    "choropleth_only": 18_683, "choropleth_only_null": 3_414,
}


def check_main(df: pd.DataFrame) -> None:
    got = facts(df)
    bad = {k: (got[k], v) for k, v in EXPECTED_MAIN.items() if got[k] != v}
    if bad:
        raise SystemExit("the main table no longer matches the CODEBOOK text -- nothing "
                         "written:\n  " + "\n  ".join(f"{k}: file {g}, text {w}"
                                                     for k, (g, w) in bad.items()))


MAIN_DOC = """# CODEBOOK

## Quick summary

- One row per map in the main table: 23,549 rows, 30 columns, one Parquet file.
- Embeddings in a separate NPZ file, one-to-one with the rows by `uid`.
- `candidates.parquet`: all 203,371 candidates of Stage 3 with the outcome of each
  step, so that the selection can be reproduced and rejected images studied.
- `validation.csv`: the human and repeat-run labels of the Technical Validation;
  summarised in `VALIDATION.md`.
- Timestamps are ISO-8601 local-naive strings; the timezone is UTC and is not written into
  the value.
- Every count in this document is computed from the published files.
- For how the corpus was selected and annotated, see README.md.

### Gates

The data descriptor numbers the steps of the cascade as Gates 1-4. In the pipeline code
Gate 1 is the size gate and Gates 2-4 are `step1`-`step3`.

| Gate | Pipeline step | Model | Decides |
|---|---|---|---|
| Gate 1 | size gate | rule, no model | shorter side of the image at least 400 px |
| Gate 2 | `step1` | Claude Haiku 4.5 | the image is a map, and the map covers at least about half of it; writes `short_description` |
| Gate 3 | `step2` | Claude Haiku 4.5 | statistical map, data referenced to administrative or statistical units, quantitative data; language of the lettering (must be English); records `has_text` |
| Gate 4 | `step3` | Claude Sonnet 4.6 | the seven thematic map types, the legend type and a confidence rating; at least one type required |

## File: `statmapcorpus_dataset.parquet`

Identical in columns and values to `statmapcorpus_dataset_v1.parquet` of version 1; only
the file name and the `dataset` entry of the file metadata changed.

### Columns

| # | Column | Type | Description and observed values |
|---|---|---|---|
| 1 | `uid` | str | 32-char lowercase hex, unique. Inherited from MapPool. Join key to the NPZ, to `candidates.parquet` and to `validation.csv`. |
| 2 | `url` | str | Image location on its original host. 18,284 https; 5,265 http. Inherited from MapPool, not normalised. |
| 3 | `domain` | str | Host from `url`. 16,374 distinct; capped at five images per domain during selection. |
| 4 | `keyword_score` | float64 | Sum of the weights of the distinct dictionary phrases matched in the MapPool alt-text; each phrase counts at most once per record. Positive phrases contribute +0.5 to +5.0, negative ones -0.5 to -5.0. Range -1.0 to 23.5; median 0.0; 39 distinct values. Records scoring below -1.0 were dropped (Stage 1). |
| 5 | `pos_keys` | str, empty where none | Pipe-separated positive phrases matched. 557 distinct here; commonest `state`, `covid`, `population`, `election`. |
| 6 | `neg_keys` | str, empty where none | Pipe-separated negative phrases matched. 140 distinct here; commonest `screen`, `city`, `image`, `urban`. |
| 7 | `pred_proba` | float64 | Stage 2 classifier probability that the image is a statistical map. 0.800015-1.0; median 0.95. The corpus was drawn from records at 0.80 and above, so this column is truncated and is not the classifier's output distribution. |
| 8 | `image_width` | int64 | Pixels. 400-16,656. |
| 9 | `image_height` | int64 | Pixels. 400-12,960. Gate 1 required `min(width, height)` of at least 400. |
| 10 | `image_max_dim` | int64 | Equals `max(width, height)` in every row. Redundant, kept for convenience. |
| 11 | `image_format` | str | JPEG 13,249; PNG 10,092; WEBP 194; BMP 14. |
| 12 | `image_size_bytes` | int64 | 6,571-20,738,937; median 104,548. |
| 13 | `sha256` | str | 64-char hex of the annotated bytes. 22,499 distinct. Exact-match check after download. |
| 14 | `phash` | str | 16-char hex, 64-bit perceptual hash. 20,399 distinct. Matches after host re-encoding, where the annotations still apply. |
| 15 | `retrieved_at` | str | Fetch time. All 1 Aug 2026, 11:19-15:26 UTC. |
| 16 | `short_description` | str | Content description written at Gate 2 by Claude Haiku 4.5. 122-557 chars; median 222; typically one or two sentences. 23 duplicated values. |
| 17 | `has_text` | bool | Gate 3: whether the map carries any lettering at all - title, legend, labels or source. `True` 23,532; `False` 17. Gate 3 required English lettering, so the 17 maps without lettering are in the corpus only because the model recorded English as their language; maps without lettering are represented only incidentally (see VALIDATION.md). |
| 18 | `method_choropleth` | bool | M1 choropleth. `True` 20,888 (88.7%). |
| 19 | `method_diagrams` | bool | M2 diagram. `True` 3,868 (16.4%). |
| 20 | `method_dot_density` | bool | M4 dot density. `True` 215 (0.9%). |
| 21 | `method_isolines` | bool | M3 isoline. `True` 275 (1.2%). |
| 22 | `method_cartogram` | bool | M6 cartogram. `True` 302 (1.3%). |
| 23 | `method_flow_map` | bool | M7 flow map. `True` 198 (0.8%). |
| 24 | `method_heat_map` | bool | M5 heat map. `True` 311 (1.3%). |
| 25 | `methods_count` | int64 | Number of thematic map types: sum of the seven `method_*` columns, verified in every row. 1 → 21,158; 2 → 2,278; 3 → 110; 4 → 2; 5 → 1. Never 0: at least one type was required for inclusion (Gate 4). |
| 26 | `legend_is_classed` | object, **null in 6,053 (25.7%)** | Gate 4. `True` 14,857: legend divided into discrete classes. `False` 2,639: continuous or unclassed legend. Null means *not applicable*: the classed/continuous distinction does not apply (no legend, a qualitative legend or an illegible one); it is not a missing value - see "Using the data". |
| 27 | `classification_confidence` | str | Gate 4: the annotating model's own confidence in the thematic map types it assigned. `high` 18,644; `medium` 4,899; `low` 6. Self-reported and not calibrated; it did not predict agreement with the human coders (VALIDATION.md). |
| 28 | `is_duplicate_byte` | bool | `True` 1,050. Marks redundant copies; one representative per group stays `False`. |
| 29 | `duplicate_group_id` | str, null in 21,663 | Equals `sha256[:16]`. 1,886 rows in 836 groups. Null means no byte-identical twin. |
| 30 | `annotated_at` | str | Gate 4 annotation time. 1 Aug 2026 15:43 - 2 Aug 2026 09:55 UTC. |

Columns 16-27 are model-generated. Their precision against two independent human coders,
their repeatability over repeated model runs and an audit of the images the cascade
rejected are reported in VALIDATION.md and in the Technical Validation section of the
data descriptor; the underlying labels are in `validation.csv`.

## Properties shared by every record

The following hold for **all 23,549 records**, by construction. They were checked per
record during annotation but are not stored as columns: a value identical in every row
states the dataset's definition rather than describing an individual map.

| Property | Value | Gate | Field in `validation.csv` |
|---|---|---|---|
| `is_map` | true | Gate 2 | - |
| `is_map_dominant` | true | Gate 2 | - |
| `is_statistical_map` | true | Gate 3 | `i2` (coders), `is_statistical_map` (retest) |
| `has_admin_units` | true: data referenced to administrative or statistical units | Gate 3 | `i3`, `has_admin_units` |
| `has_quantitative_data` | true | Gate 3 | `i4`, `has_quantitative_data` |
| `map_language` | `en` | Gate 3 | `map_language` (retest) |
| `methods_count` | ≥ 1: at least one thematic map type | Gate 4 | - |

Anything failing one of these was excluded before the corpus was formed, so this file
cannot be used to study how often maps fail them; `candidates.parquet` can. The
values are also written into the parquet file's key-value metadata, so they travel with
the data:

```python
import pyarrow.parquet as pq
meta = pq.read_schema("statmapcorpus_dataset.parquet").metadata
meta[b"inclusion_criteria"], meta[b"source_corpus"], meta[b"pipeline_version"]
```

### Thematic map types (M1-M7)

Multi-label: the seven flags are independent and multi-type maps are common. The typology
follows cartographic textbooks; the definitions below are those of Table 3 of the data
descriptor and summarise the instructions given to the annotating model at Gate 4 (full
wording in `src/03_04_retrieval_and_cascade/multiturn/prompts/step3.md` of the code
repository). The column names keep the prefix `method_` and the name `methods_count` of
version 1; each `method_*` column holds one thematic map type label.

| ID | Thematic map type | Column | Definition used for labelling |
|---|---|---|---|
| M1 | Choropleth | `method_choropleth` | Areal units filled with colour, shading or hatching that varies with the mapped value; excludes uniformly filled reference maps. |
| M2 | Diagram | `method_diagrams` | Symbols or small charts placed on the map to encode magnitude: proportional, graduated and structural symbols, pie and bar charts; excludes constant-size markers. |
| M3 | Isoline | `method_isolines` | Contours of equal value, as lines or as filled bands with a classed legend, whose boundaries do not follow administrative or statistical units. |
| M4 | Dot density | `method_dot_density` | Many small uniform dots scattered within areas, each representing a fixed count. |
| M5 | Heat map | `method_heat_map` | Continuous, smoothly blending colour surface (raster, kernel density or gridded) without class boundaries. |
| M6 | Cartogram | `method_cartogram` | Geometry of units visibly deformed so that their size encodes a data value. |
| M7 | Flow map | `method_flow_map` | Directed lines, arrows or curves connecting locations to represent movement, often with width encoding magnitude. |

Maps matching none of the seven were rejected at Gate 4 and are not in the corpus (97
images; see `candidates.parquet`).

## File: `embeddings_clip_vitl14.npz`

Unchanged from version 1.

`uids` - 23,549 strings, same values and same order as the `uid` column.
`emb` - float32, shape (23,549, 768), L2-normalised at source (observed norms
1.0000 ± 0.0006), no NaN or infinite values.

These are OpenAI CLIP ViT-L/14 image features (`open_clip`: model `ViT-L-14`, pretrained
tag `openai`; Hugging Face: `openai/clip-vit-large-patch14`), computed by DataComp for
CommonPool and passed through MapPool unchanged. Standard CLIP preprocessing applies:
bicubic resize, 224×224 centre crop, OpenAI channel means and standard deviations.

Note that CommonPool resized images so that the largest dimension did not exceed 512 px
*before* extracting features. Images in this corpus run up to 16,656 px wide, so the
embedding encodes a heavily downscaled view. Fine detail - legend text, thin borders,
small labels - is unlikely to survive into the vector, which matters if you intend to use
these embeddings to detect textual or fine-grained cartographic properties.
"""

USING_DOC = """
## Using the data

**Group on `phash`, not on `duplicate_group_id`, when building splits.** Columns 28-29 of
the main table cover byte-identical files only. 5,180 rows share a `phash` with at least
one other row, but only 1,886 of them carry a `duplicate_group_id`: the remaining 3,294
are re-encodings and rescalings that `sha256` grouping cannot see. Relaxing the match to a
Hamming distance of 2, which catches typical re-encodings, raises the affected total to
6,310 rows. Partitioning on `duplicate_group_id` alone will leak between train and test.

**Report macro-averaged metrics, not accuracy.** The thematic map type labels are
multi-label and severely imbalanced: choropleth covers 88.7% of records while the five
rarest types each sit below 1.5%. Precision also differs by type: it was high for
choropleths, diagrams and cartograms, lower for heat maps, dot density and flow maps and
lowest for isolines (VALIDATION.md), so per-type results on the rare types carry label
noise.

**Do not impute null (*not applicable*) in `legend_is_classed` as `False`.** The annotating
model was instructed to return null when the classed-vs-continuous distinction does not apply: no
legend is present, the legend is qualitative (unordered hues for nominal categories)
rather than a quantitative scale, or it is illegible. Ties were to be broken toward `True`,
so null is not a catch-all for uncertainty.

The three causes are not separable in this release. Their relative weight can be read
indirectly: the null rate is flat across resolution quartiles (24.6-27.9%), so
illegibility is not the main driver, but it varies sharply by type: 2.2% for isolines,
20.6% for choropleth, 45-60% for diagram maps, flow maps, dot density, heat maps and
cartograms, where value is carried by symbols rather than by a classed fill. Among
choropleth-only maps, where a classed legend would be expected, 3,414 of 18,683 (18.3%)
are still null.

Null is not missing at random; it correlates with type and with
`classification_confidence` (16.3% of `high`, 61.5% of `medium`), so a complete-case
analysis over the 17,496 resolved rows is biased toward maps with a legible quantitative
legend. Report that denominator explicitly. In the Parquet file the column is a boolean
with nulls; pandas reads it as `object` (Python `None` mixed with `bool`) unless it is read
with `pd.read_parquet(..., dtype_backend="numpy_nullable")`.
"""


CANDIDATE_DESCR = {
    "uid": "MapPool record identifier; joins to the main table for the 23,549 corpus maps",
    "url": "image location on its original host, as inherited from MapPool",
    "domain": "host from `url`; at most five candidates per domain",
    "keyword_score": "Stage 1 dictionary score (see the main table)",
    "pred_proba": "Stage 2 classifier probability; 0.80 and above by selection",
    "url_status": "HTTP HEAD check on 30 July or 1 August 2026: `alive`, `dead` (error status or no "
                  "response) or `error` (request could not be made)",
    "url_http_code": "HTTP status code of the check; null when no response was received",
    "download_status": "download on 1 August 2026: `success`, `not_image` (the file could "
                       "not be decoded as an image) or `failed`; null when the URL check "
                       "failed and no download was attempted",
    "image_width": "pixels; null unless `download_status` = `success`",
    "image_height": "pixels; null unless `download_status` = `success`",
    "image_format": "decoded file format; null unless `download_status` = `success`",
    "gate1_pass": "Gate 1: shorter side at least 400 px",
    "gate2_is_map": "Gate 2: the image shows a map",
    "gate2_map_dominant": "Gate 2: the map, with its legend and marginal elements, covers "
                          "at least about half of the image",
    "gate2_pass": "Gate 2 passed: `gate2_is_map` and `gate2_map_dominant`",
    "gate3_is_statistical": "Gate 3: statistical map",
    "gate3_admin_units": "Gate 3: data referenced to administrative or statistical units",
    "gate3_quantitative": "Gate 3: quantitative data",
    "gate3_has_text": "Gate 3: the map carries any lettering (recorded, not a criterion)",
    "gate3_language": "Gate 3: dominant language of the lettering, ISO 639-1 code; "
                      "`unknown` when the model could not assign one (including maps "
                      "without lettering), `other` for a language outside its list",
    "gate3_pass": "Gate 3 passed: the three content conditions and `gate3_language` = `en`",
    "gate4_n_types": "Gate 4: number of thematic map types assigned (0-7)",
    "gate4_pass": "Gate 4 passed: at least one type",
    "in_corpus": "the record is in `statmapcorpus_dataset.parquet`",
}


def domain_of(s: pd.Series, name: str = "") -> str:
    nn = s.dropna()
    if nn.empty:
        return "(all null)"
    if str(s.dtype) in ("bool", "boolean"):
        t = int((nn == True).sum())
        return f"true {t:,}; false {len(nn) - t:,}"
    if pd.api.types.is_numeric_dtype(nn):
        if nn.nunique() <= 8:
            vc = nn.value_counts().sort_index()
            return "; ".join(f"{k} → {v:,}" for k, v in vc.items())
        if pd.api.types.is_integer_dtype(nn):
            return f"{int(nn.min()):,} .. {int(nn.max()):,} (median {nn.median():,.0f})"
        return f"{nn.min():.4g} .. {nn.max():.4g} (median {nn.median():.4g})"
    u = nn.unique()
    if len(u) <= 8:
        vc = nn.value_counts()
        return "; ".join(f"`{k}` {v:,}" for k, v in vc.items())
    ex = str(nn.iloc[0])
    if name in {"uid"} or len(ex) <= 50:
        return f"{len(u):,} distinct, e.g. `{ex}`"
    return f"{len(u):,} distinct, e.g. `{ex[:47]}…`"


def candidates_doc(df: pd.DataFrame) -> str:
    rows = ["## File: `candidates.parquet`", "",
            f"**{len(df):,} rows × {len(df.columns)} columns.** One row for every record "
            "selected in Stage 3 (classifier probability at least 0.80, at most five images "
            "per domain), with the outcome of each step from the URL check to Gate 4. A "
            "gate column is null when the record never reached that gate, so the gates can "
            "be read as a funnel. Raw model responses, descriptions of rejected images and "
            "local file paths are not included; the labels of the corpus maps are in the "
            "main table.", "",
            "| Column | Type | Non-null | Values | Description |",
            "|---|---|---:|---|---|"]
    for c in df.columns:
        rows.append(f"| `{c}` | {df[c].dtype} | {int(df[c].notna().sum()):,} | "
                    f"{domain_of(df[c], c)} | {CANDIDATE_DESCR.get(c, '')} |")

    content_ok = (df.gate3_is_statistical & df.gate3_admin_units
                  & df.gate3_quantitative).fillna(False)
    language_only = content_ok & df.gate3_language.ne("en")
    lang = df.loc[language_only, "gate3_language"]
    no_text = int((language_only & (df.gate3_language == "unknown")
                   & (df.gate3_has_text == False)).sum())
    step = [
        ("Stage 3 candidates", len(df), ""),
        ("URL alive", int((df.url_status == "alive").sum()),
         f"dead {int((df.url_status == 'dead').sum()):,}; error {int((df.url_status == 'error').sum()):,}"),
        ("Downloaded and decoded", int((df.download_status == "success").sum()),
         f"not an image {int((df.download_status == 'not_image').sum()):,}; failed {int((df.download_status == 'failed').sum()):,}"),
        ("Gate 1 passed", int((df.gate1_pass == True).sum()),
         f"too small {int((df.gate1_pass == False).sum()):,}"),
        ("Gate 2 passed", int((df.gate2_pass == True).sum()),
         f"not a map {int((df.gate2_is_map == False).sum()):,}; map not dominant "
         f"{int(((df.gate2_is_map == True) & (df.gate2_map_dominant == False)).sum()):,}"),
        ("Gate 3 passed", int((df.gate3_pass == True).sum()),
         f"content {int(((df.gate3_pass == False) & ~content_ok).sum()):,}; language alone "
         f"{int(language_only.sum()):,}"),
        ("Gate 4 passed = corpus", int((df.gate4_pass == True).sum()),
         f"no type {int((df.gate4_pass == False).sum()):,}"),
    ]
    rows += ["", "### Reading the funnel", "",
             "| Step | Remaining | Rejected |", "|---|---:|---|"]
    rows += [f"| {a} | {b:,} | {c} |" for a, b, c in step]
    rows += ["",
             f"Of the {int(language_only.sum()):,} images rejected at Gate 3 for language "
             f"alone, {int((lang != 'unknown').sum()):,} were assigned another language "
             f"and {int((lang == 'unknown').sum()):,} `unknown`; {no_text:,} of the latter "
             "carry no lettering at all.", "",
             "Example - every image that met all content conditions of Gate 3 but was "
             "rejected for its language:", "",
             "```python",
             "c = pd.read_parquet(\"candidates.parquet\")",
             "ok = c.gate3_is_statistical & c.gate3_admin_units & c.gate3_quantitative",
             "rejected_for_language = c[ok.fillna(False) & c.gate3_language.ne(\"en\")]",
             "```", "",
             "The images themselves are not distributed. `download_images.py` reads the "
             "`uid` and `url` columns, so it can fetch any subset of this file, but no "
             "checksum is recorded for the rejected images: there is nothing to verify a "
             "later download against.", ""]
    return "\n".join(rows)


def validation_doc(v: pd.DataFrame) -> str:
    by_block = []
    for block in ("coders", "retest", "audit"):
        g = v[v.block == block]
        by_block.append(f"| `{block}` | {len(g):,} | {g.uid.nunique():,} | "
                        f"{', '.join(f'`{x}`' for x in sorted(g.source.unique()))} | "
                        f"{', '.join(f'`{x}`' for x in sorted(g.field.unique()))} |")
    strata = v[v.block != "coders"].groupby(["block", "stratum"]).uid.nunique()
    strata_txt = "; ".join(f"{b} `{s}` {n}" for (b, s), n in strata.items())
    return f"""
## File: `validation.csv`

**{len(v):,} rows.** The published record of the three validation exercises of the
Technical Validation, one row per observation (long format). VALIDATION.md summarises the
results; the scripts that reproduce every number from this file and the main table are in
the `validation/` directory of the code repository.

| Block | Rows | Maps | `source` | `field` |
|---|---:|---:|---|---|
{chr(10).join(by_block)}

Maps per stratum: {strata_txt}.

| Column | Description |
|---|---|
| `block` | `coders`: two independent human coders on the 157-map validation sample. `retest`: two repeated runs of Gates 3 and 4 on 300 corpus maps. `audit`: blind coding of 130 images rejected by the cascade and 20 corpus maps as controls. |
| `uid` | MapPool record identifier. Joins to the main table (coders, retest, audit controls) or to `candidates.parquet` (audit). |
| `stratum` | `coders`: empty, because a map enters the stratified sample through any of its labels; see `inclusion_probability`. `retest`: `validation` (the 157 validation maps) or `random` (143 maps drawn at random from the rest of the corpus). `audit`: `G2_not_a_map`, `G2_not_dominant`, `G3_content`, `G3_language`, `G4_no_type` (the gate and reason of rejection) or `DECOY` (corpus maps added as controls). |
| `source` | `coders`: `coder1` or `coder2`. `retest`: `run1` or `run2`; the production labels are the main table. `audit`: `coder1`. |
| `field` | `coders`: `i2` statistical map, `i3` administrative or statistical units, `i4` quantitative data, `has_text`, `m1`-`m7` (the thematic map types, numbered as in this codebook), `legend_is_classed`. `retest`: the same type and legend fields and `has_text`, plus `is_statistical_map`, `has_admin_units`, `has_quantitative_data`, `map_language`. `audit`: `meets_all_criteria`. |
| `value` | `coders` and `audit`: `yes`, `no` or `cannot_assess`; for `legend_is_classed`: `classed`, `continuous` or `other` (not applicable: no legend, a qualitative one or an illegible one; null in the main table). `retest`: `true` or `false`; `classed`, `continuous` or `other`; a language code for `map_language`. |
| `inclusion_probability` | `coders` only: probability that the map entered the stratified sample; its reciprocal is the Horvitz-Thompson weight. |
| `population_n` | `audit` only: size of the stratum among the rejected images (for `DECOY`, the corpus maps eligible as controls), needed to extrapolate. |
| `response_time_s` | `audit` only: seconds the coder spent on the map. |

`coder1` is the author of the dataset, who designed the pipeline; `coder2` an independent
coder. Neither saw model output while coding. Disputed judgements are kept as recorded;
the three reference standards built from them (A, B, C) are explained in VALIDATION.md.
"""


def main() -> None:
    for path in (MAIN, CANDIDATES, VALIDATION):
        if not path.exists():
            raise SystemExit(f"{path} not found -- run the exports first")
    main_df = pd.read_parquet(MAIN)
    check_main(main_df)
    cand = pd.read_parquet(CANDIDATES)
    if int(cand.in_corpus.sum()) != len(main_df) or not set(
            cand.loc[cand.in_corpus, "uid"]) == set(main_df.uid):
        raise SystemExit("candidates.parquet: in_corpus does not match the main table")
    val = pd.read_csv(VALIDATION, dtype=str)

    text = MAIN_DOC + "\n" + candidates_doc(cand) + validation_doc(val) + USING_DOC
    (BASE / "CODEBOOK.md").write_text(text, encoding="utf-8")
    print(f"CODEBOOK.md        {len(text.splitlines())} lines  ->  {BASE / 'CODEBOOK.md'}")


if __name__ == "__main__":
    main()
