# Analysis plans

The three validation exercises of the data descriptor followed analysis plans that were
written and dated before their data were collected. This file condenses those plans to the
elements on which the reported analyses rest. Operational details and secondary analyses
that the data descriptor does not report are left out. The original plans, written in
Polish, are available from the author on request.

| Plan | Recorded | Status at recording |
|---|---|---|
| Coder validation | 30 August 2026 | before coding began |
| Repeatability | 2 October 2026 | before any model call |
| Rejection audit | 3 October 2026, amended the same day | both before coding began |

## 1. Coder validation

**Question.** How precise are the model's labels for the seven thematic map types
(M1-M7), lettering (`has_text`) and legend type (`legend_is_classed`: classed, continuous,
not applicable)?

**Sample.** 157 corpus maps, stratified by the model's verdict: 20 maps for each of M1-M7
and all 17 maps labelled without lettering. Seed 42. Byte duplicates are excluded, leaving
a frame of 22,499 maps. Each stratum is drawn from all maps carrying its label, so hybrid
maps enter in their natural proportion. Assigning each map to its rarest label
was rejected, because it would remove hybrids from the choropleth stratum and inflate its
precision.

**Reference standard.** Two coders code all 157 maps independently, each in a different
order. Agreed answers form the reference. Disagreements were to be arbitrated by the
supervisor. A disagreement rate above 10% for any field is reported as a result. Answers
"cannot assess" leave the denominator of the field concerned. Questions I2-I4 (statistical
map, administrative or statistical units, quantitative data) are not validated as labels.
The share of maps meeting all three estimates the purity of the frame, and maps that the
coders agree fail one of them leave the analysis of types.

**Weighting and intervals.** A map can enter the sample through any of its labels, so
π = 1 − ∏ₛ(1 − kₛ/|poolₛ|). Precision is estimated with Horvitz-Thompson weights 1/π.
Intervals are Wilson intervals on the Kish effective sample size.

**Measures.** Per field:

- precision of the positive label (the headline measure);
- Cohen's κ between model and reference;
- Cohen's κ between the two coders;
- raw agreement alongside κ wherever marginals are extreme.

Recall is not estimated, by design. The sample is drawn from accepted maps.

## 2. Repeatability

**Question.** What share of the labels would change if Gates 3 and 4 were run again? Are
unstable labels more often wrong?

**Sample.** 300 corpus maps: the 157 validation maps plus 143 maps drawn at random (seed 42)
from the remaining 22,342 non-duplicate corpus maps. Only the random layer describes the
corpus. The validation layer is used solely to link instability to error.

**Runs.** The production labels plus two new runs, each with the production code,
prompts, models (Claude Haiku 4.5 at Gate 3, Claude Sonnet 4.6 at Gate 4), output limits
and default temperature. If a model is unavailable, the study stops; no other model is
substituted. Earlier gates' responses, which are part of the input, are fixed at their
production values. The measured instability is therefore a lower bound for a full
re-run. Gate 4 runs for all 300 maps, whatever the fresh Gate 3 says.

**Measures.**

- *Gate 4, per type and legend:* raw agreement, Cohen's κ for each pair of runs, Fleiss's
  κ, and the share stable across all three runs.
- *Gate 4, whole label set:* the share of maps whose complete set of seven types, with and
  without the legend, is identical in all three runs.
- *Gate 3:* the share of maps that would fail the gate in a fresh run, per condition and
  overall, with Wilson intervals.
- *Confidence:* stability by the model's declared confidence.
- *Instability and error (validation layer):* units are (map, field) pairs, against
  reference C (coder agreement only). The 2×2 table of stable/unstable against
  right/wrong is reported with the odds ratio and Fisher's exact test. The directional
  hypothesis is that unstable labels are more often wrong. The test is exploratory,
  because pairs within a map are not independent.

**Unfavourable result.** If fewer than about 90% of random-layer maps keep their complete
type set, this is reported as a limitation of the dataset.

## 3. Rejection audit

**Question.** How many images rejected by the cascade should have been in the corpus?

**Strata.** The rejection strata and their populations are:

| Stratum | Population |
|---|---|
| Gate 2, not a map | 337 |
| Gate 2, map not dominant | 425 |
| Gate 3, content criteria not met | 25,984 |
| Gate 3, lettering not English | 17,643 |
| Gate 4, no thematic map type | 97 |

The audit also includes corpus maps as decoys (`DECOY`), drawn from 22,342 non-duplicate
corpus maps outside the validation sample. Sampling within strata uses seed 42. Gate 1 is
a mechanical size filter and is not audited. Images lost before download cannot be shown
and are not audited.

**Blindness.** One coder codes every image. She does not see the stratum, the model's
verdict or its description. Strata and decoys are mixed in one random order.

**Amendment 1 (before coding).** The amendment changed three things:

- *Sample:* reduced from 190 to 150 images (15, 15, 60, 20, 20 and 20 decoys). Each stratum
  takes the first images of the same permutation, and the presentation order was redrawn
  with seed 43.
- *Question:* seven questions were replaced by one: "Does the image meet all the
  StatMapCorpus criteria?" (yes / no / cannot assess). A wrongful rejection is therefore
  judged against all corpus criteria, not only those of the rejecting gate.
- *What is lost:* the per-criterion breakdown and the field-by-field comparison with the
  model.

**Measures.** For each stratum, the share of "yes" answers, with a 95% Wilson interval and
an extrapolation (rate × population, with the interval carried over). Answers "cannot
assess" leave the denominator and their number is reported.

**Decoy control and unfavourable result.** Fixed in advance:

- If fewer than about 70% of decoys are judged to meet all criteria (14 of 20), the
  coding is too strict. The rejection rates are then treated as overstated and reported
  as a limitation.
- A wrongful-rejection rate above about 20% in the Gate 3 content stratum is reported as a
  limitation of the dataset.
- The results rest on one coder and are described as such.

## Departures from the plans

Both concern the coder validation.

- Disagreements between the coders were not arbitrated. Precision is instead reported
  under three reference standards: disputes resolved in favour of the first coder (A), in
  favour of the second (B), or excluded (C).
- The 9 of 157 maps that both coders judged to fail one of I2-I4 were not removed from the
  analysis of types, so precision refers to all sampled corpus maps. Removing them changes
  no precision by more than 6.3 percentage points under any reference standard. Under C,
  isoline maps would be 0.46 instead of 0.52, dot density 0.73 instead of 0.67 and heat
  maps 0.67 instead of 0.71; every other label changes by less than one point.
