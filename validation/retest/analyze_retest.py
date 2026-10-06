import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coders"))

from agreement import (CODERS, MISSING, cohen_kappa, find_main_table,
                       find_validation, load_codings)

RUNS = ("prod", "run1", "run2")
LAYERS = ("random", "validation", "all")

METHOD_COLUMN = {
    "m1": "method_choropleth",
    "m2": "method_diagrams",
    "m3": "method_isolines",
    "m4": "method_dot_density",
    "m5": "method_heat_map",
    "m6": "method_cartogram",
    "m7": "method_flow_map",
}
METHODS = tuple(METHOD_COLUMN)
METHOD_LABEL = {
    "m1": "M1 · choropleth", "m2": "M2 · diagram map", "m3": "M3 · isarithmic",
    "m4": "M4 · dot density", "m5": "M5 · heat map", "m6": "M6 · cartogram",
    "m7": "M7 · flow map",
}
LEGEND_MAP = {True: "classed", False: "continuous", None: "other"}
GATE3_CONSTANT = ("is_statistical_map", "has_admin_units", "has_quantitative_data")
GATE3_FIELDS = GATE3_CONSTANT + ("map_language", "has_text")
Z = 1.959963985


def wilson(p, n):
    if n <= 0 or p is None or not 0.0 <= p <= 1.0:
        return None, None
    d = 1.0 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z / d * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def fleiss_kappa(rows):
    rows = [r for r in rows if len(r) >= 2]
    if not rows:
        return None
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        return None
    categories = sorted({v for r in rows for v in r})
    counts = np.array([[r.count(c) for c in categories] for r in rows], dtype=float)
    p_i = (np.square(counts).sum(axis=1) - n) / (n * (n - 1))
    p_bar = float(p_i.mean())
    p_j = counts.sum(axis=0) / (len(rows) * n)
    p_e = float(np.square(p_j).sum())
    if np.isclose(p_e, 1.0):
        return None
    return (p_bar - p_e) / (1.0 - p_e)


def pair_stats(records, left, right):
    pairs = [(r[left], r[right]) for r in records]
    if not pairs:
        return {"n": 0, "agreement": None, "kappa": None}
    agree = sum(1 for a, b in pairs if a == b) / len(pairs)
    k = cohen_kappa(pairs)
    return {"n": len(pairs), "agreement": agree,
            "kappa": None if k != k else float(k)}


def stability_block(records, field):
    rows = [[str(r[run]) for run in RUNS] for r in records]
    stable = sum(1 for r in rows if len(set(r)) == 1)
    n = len(rows)
    low, high = wilson(stable / n, n) if n else (None, None)
    block = {
        "field": field,
        "n": n,
        "stable_all3": stable,
        "stable_all3_share": stable / n if n else None,
        "stable_ci95": [low, high],
        "fleiss_kappa": fleiss_kappa(rows),
        "pairs": {},
        "distribution_prod": {},
    }
    plain = [{run: str(r[run]) for run in RUNS} for r in records]
    for a, b in (("prod", "run1"), ("prod", "run2"), ("run1", "run2")):
        block["pairs"][f"{a}_vs_{b}"] = pair_stats(plain, a, b)
    distribution = pd.Series([r["prod"] for r in plain]).value_counts()
    block["distribution_prod"] = {str(k): int(v) for k, v in distribution.items()}
    return block


def load_retest(path: Path) -> tuple[dict, dict]:
    values, layer = {}, {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["block"] != "retest":
                continue
            values[(row["source"], row["uid"], row["field"])] = row["value"]
            layer[row["uid"]] = row["stratum"]
    return values, layer


def load_production(deposit: Path, uids: set) -> tuple[pd.DataFrame, dict]:
    table = find_main_table(deposit)
    frame = pd.read_parquet(table)
    frame = frame[frame["uid"].isin(uids)].set_index("uid")
    metadata = pq.read_schema(table).metadata or {}
    raw = metadata.get(b"inclusion_criteria")
    if raw is None:
        raise SystemExit(f"{table.name} has no `inclusion_criteria` metadata; "
                         "the production gate-3 values cannot be read from the deposit")
    criteria = json.loads(raw.decode("utf-8"))
    for field in GATE3_CONSTANT:
        if criteria.get(field) is not True:
            raise SystemExit(f"metadata says {field} = {criteria.get(field)!r}, "
                             "expected true for every record")
    return frame, criteria


def norm_bool(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return {"true": True, "false": False}.get(value.lower())
    if isinstance(value, float) and value != value:
        return None
    return bool(value)


def legend_cat(value):
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return "other"
    if isinstance(value, str):
        return value if value in LEGEND_MAP.values() else "other"
    return LEGEND_MAP[bool(value)]


class Labels:

    def __init__(self, production: pd.DataFrame, criteria: dict, retest: dict):
        self.production = production
        self.criteria = criteria
        self.retest = retest

    def get(self, run, uid, field):
        if run != "prod":
            return self.retest.get((run, uid, field))
        if field in METHOD_COLUMN:
            return self.production.at[uid, METHOD_COLUMN[field]]
        if field in ("legend_is_classed", "has_text"):
            return self.production.at[uid, field]
        if field in GATE3_CONSTANT:
            return True
        if field == "map_language":
            return self.criteria["map_language"]
        raise KeyError(field)

    def boolean(self, run, uid, field):
        return norm_bool(self.get(run, uid, field))

    def legend(self, run, uid):
        return legend_cat(self.get(run, uid, "legend_is_classed"))

    def language(self, run, uid):
        return str(self.get(run, uid, "map_language") or "").lower()


def layer_uids(layer: dict, labels: Labels, cut: str) -> list:
    selected = [u for u, name in sorted(layer.items())
                if cut == "all" or name == cut]
    complete = []
    for uid in selected:
        if uid not in labels.production.index:
            continue
        if all(labels.retest.get((run, uid, field)) is not None
               for run in ("run1", "run2")
               for field in METHODS + ("legend_is_classed", "has_text",
                                       "map_language")):
            complete.append(uid)
    return complete


def gate4_section(layer, labels):
    out = {}
    for cut in LAYERS:
        uids = layer_uids(layer, labels, cut)
        body = {"n": len(uids), "fields": {}}

        for field in METHODS:
            records = [{run: labels.boolean(run, u, field) for run in RUNS}
                       for u in uids]
            block = stability_block(records, field)
            block["label"] = METHOD_LABEL[field]
            body["fields"][field] = block

        records = [{run: labels.legend(run, u) for run in RUNS} for u in uids]
        block = stability_block(records, "legend_is_classed")
        block["label"] = "legend (classed/continuous/other)"
        body["fields"]["legend_is_classed"] = block

        def complete(uid, with_legend):
            keys = []
            for run in RUNS:
                row = tuple(labels.boolean(run, uid, f) for f in METHODS)
                if with_legend:
                    row = row + (labels.legend(run, uid),)
                keys.append(row)
            return len(set(keys)) == 1

        for name, with_legend in (("methods7", False),
                                  ("methods7_plus_legend", True)):
            hits = sum(1 for u in uids if complete(u, with_legend))
            share = hits / len(uids) if uids else None
            low, high = wilson(share, len(uids)) if uids else (None, None)
            body[f"stable_{name}"] = {"n": len(uids), "stable": hits,
                                      "share": share, "ci95": [low, high]}
        out[cut] = body
    return out


def label_repeatability(layer, labels):
    out = {}
    for cut in LAYERS:
        uids = layer_uids(layer, labels, cut)
        body = {}
        for field in METHODS:
            positives = [u for u in uids if labels.boolean("prod", u, field)]
            stable = sum(1 for u in positives
                         if len({labels.boolean(run, u, field) for run in RUNS}) == 1)
            share = stable / len(positives) if positives else None
            low, high = wilson(share, len(positives)) if positives else (None, None)
            body[field] = {"label": METHOD_LABEL[field], "n_positive": len(positives),
                           "stable_all3": stable, "share": share,
                           "ci95": [low, high]}
        out[cut] = body
    return out


def gate3_section(layer, labels):
    def passes(run, uid):
        return (bool(labels.boolean(run, uid, "is_statistical_map"))
                and bool(labels.boolean(run, uid, "has_admin_units"))
                and bool(labels.boolean(run, uid, "has_quantitative_data"))
                and labels.language(run, uid) == "en")

    out = {}
    for cut in LAYERS:
        uids = layer_uids(layer, labels, cut)
        body = {"n": len(uids), "runs": {}}

        for run in ("run1", "run2"):
            fails = {"total": 0, "is_statistical_map": 0, "has_admin_units": 0,
                     "has_quantitative_data": 0, "language": 0}
            for uid in uids:
                if passes(run, uid):
                    continue
                fails["total"] += 1
                for field in GATE3_CONSTANT:
                    if not labels.boolean(run, uid, field):
                        fails[field] += 1
                if labels.language(run, uid) != "en":
                    fails["language"] += 1
            share = fails["total"] / len(uids) if uids else None
            low, high = wilson(share, len(uids)) if uids else (None, None)
            body["runs"][run] = {
                **fails, "share_total": share, "ci95_total": [low, high],
                "shares": {k: (v / len(uids) if uids else None)
                           for k, v in fails.items() if k != "total"}}

        either = sum(1 for u in uids
                     if not passes("run1", u) or not passes("run2", u))
        share = either / len(uids) if uids else None
        low, high = wilson(share, len(uids)) if uids else (None, None)
        body["fail_in_either_run"] = {"n": either, "share": share,
                                      "ci95": [low, high]}

        for field in GATE3_CONSTANT + ("has_text",):
            records = [{run: labels.boolean(run, u, field) for run in RUNS}
                       for u in uids]
            body[field] = stability_block(records, field)
        records = [{run: labels.language(run, u) for run in RUNS} for u in uids]
        body["map_language"] = stability_block(records, "map_language")

        textless = {}
        for run in ("run1", "run2"):
            selected = [u for u in uids
                        if not labels.boolean(run, u, "has_text")
                        and labels.language(run, u) != "en"]
            textless[run] = {"n": len(selected), "uids": selected}
        union = sorted(set(textless["run1"]["uids"]) | set(textless["run2"]["uids"]))
        textless["in_at_least_one_run"] = {"n": len(union), "uids": union}
        textless["production_textless_n"] = sum(
            1 for u in uids if labels.boolean("prod", u, "has_text") is False)
        body["textless_and_language_other_than_en"] = textless

        out[cut] = body
    return out


def confidence_section(layer, labels, confidence):
    out = {}
    for cut in LAYERS:
        uids = layer_uids(layer, labels, cut)
        body = {}
        for level in ("high", "medium", "low"):
            selected = [u for u in uids if confidence.get(u) == level]
            if not selected:
                body[level] = {"n": 0}
                continue
            hits = 0
            for uid in selected:
                keys = {tuple(labels.boolean(run, uid, f) for f in METHODS)
                        for run in RUNS}
                hits += len(keys) == 1
            share = hits / len(selected)
            low, high = wilson(share, len(selected))
            body[level] = {
                "n": len(selected), "stable": hits,
                "share": share if len(selected) >= 10 else None,
                "ci95": [low, high] if len(selected) >= 10 else None,
                "note": None if len(selected) >= 10 else "n < 10 - raw counts only"}
        out[cut] = body
    return out


def gold_section(layer, labels, codings):
    uids = layer_uids(layer, labels, "validation")
    fields = list(METHODS) + ["legend_is_classed"]
    rows = []
    for field in fields:
        for uid in uids:
            left = codings.get((CODERS[0], uid, field))
            right = codings.get((CODERS[1], uid, field))
            if left is None or right is None or MISSING in (left, right) \
                    or left != right:
                continue
            gold = left
            if field == "legend_is_classed":
                values = {run: labels.legend(run, uid) for run in RUNS}
            else:
                values = {run: ("yes" if labels.boolean(run, uid, field) else "no")
                          for run in RUNS}
            rows.append({"uid": uid, "field": field,
                         "unstable": len(set(values.values())) > 1,
                         "wrong": values["prod"] != gold, "gold": gold,
                         **{f"value_{run}": values[run] for run in RUNS}})

    pairs = pd.DataFrame(rows)
    out = {"n_pairs": len(pairs), "n_maps": len(uids),
           "coders": list(CODERS), "variant": "C"}
    if pairs.empty:
        return out, pairs

    table = pd.crosstab(pairs.unstable, pairs.wrong).reindex(
        index=[False, True], columns=[False, True], fill_value=0)
    a, b = int(table.loc[False, False]), int(table.loc[False, True])
    c, d = int(table.loc[True, False]), int(table.loc[True, True])
    out["table_2x2"] = {"stable_right": a, "stable_wrong": b,
                        "unstable_right": c, "unstable_wrong": d}
    out["error_among_stable"] = b / (a + b) if (a + b) else None
    out["error_among_unstable"] = d / (c + d) if (c + d) else None
    n_unstable = c + d
    if n_unstable >= 10:
        _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        out["odds_ratio_error_if_unstable"] = (float((d * a) / (c * b))
                                               if c and b else None)
        out["fisher_p_two_sided"] = float(p)
    else:
        out["note"] = (f"only {n_unstable} unstable pairs (< 10) - "
                       "no odds ratio and no test")
    out["per_field"] = {}
    for field, group in pairs.groupby("field"):
        unstable = group[group.unstable]
        out["per_field"][field] = {
            "n": len(group),
            "unstable": int(group.unstable.sum()),
            "wrong": int(group.wrong.sum()),
            "error_among_stable": float((~group.unstable & group.wrong).sum()
                                        / max((~group.unstable).sum(), 1)),
            "error_among_unstable": (float(unstable.wrong.mean())
                                     if len(unstable) else None),
        }
    out["caveats"] = [
        "the 157-map layer is stratified by the model's verdict - the error rate "
        "is not the error rate of the corpus",
        "(map, field) pairs are not independent within a map; the test is exploratory",
    ]
    return out, pairs


def build_csv(results):
    rows = []
    for cut, body in results["gate4"].items():
        for field, block in body["fields"].items():
            rows.append({
                "section": "gate4", "layer": cut, "field": field,
                "label": block.get("label", field), "n": block["n"],
                "stable_3runs": block["stable_all3"],
                "stable_share": block["stable_all3_share"],
                "fleiss_kappa": block["fleiss_kappa"],
                "agreement_prod_run1": block["pairs"]["prod_vs_run1"]["agreement"],
                "kappa_prod_run1": block["pairs"]["prod_vs_run1"]["kappa"],
                "agreement_prod_run2": block["pairs"]["prod_vs_run2"]["agreement"],
                "kappa_prod_run2": block["pairs"]["prod_vs_run2"]["kappa"],
                "agreement_run1_run2": block["pairs"]["run1_vs_run2"]["agreement"],
                "kappa_run1_run2": block["pairs"]["run1_vs_run2"]["kappa"],
            })
        for name in ("methods7", "methods7_plus_legend"):
            block = body[f"stable_{name}"]
            rows.append({"section": "gate4_complete_set", "layer": cut,
                         "field": name, "label": name, "n": block["n"],
                         "stable_3runs": block["stable"],
                         "stable_share": block["share"]})
    for cut, body in results["label_repeatability"].items():
        for field, block in body.items():
            rows.append({"section": "label_repeatability", "layer": cut,
                         "field": field, "label": block["label"],
                         "n": block["n_positive"],
                         "stable_3runs": block["stable_all3"],
                         "stable_share": block["share"]})
    for cut, body in results["gate3"].items():
        for run, block in body["runs"].items():
            rows.append({"section": "gate3_rejected", "layer": cut, "field": run,
                         "label": "would not pass the gate", "n": body["n"],
                         "stable_3runs": block["total"],
                         "stable_share": block["share_total"]})
        for field in GATE3_FIELDS:
            block = body[field]
            rows.append({
                "section": "gate3_stability", "layer": cut, "field": field,
                "label": field, "n": block["n"],
                "stable_3runs": block["stable_all3"],
                "stable_share": block["stable_all3_share"],
                "fleiss_kappa": block["fleiss_kappa"],
                "agreement_prod_run1": block["pairs"]["prod_vs_run1"]["agreement"],
                "kappa_prod_run1": block["pairs"]["prod_vs_run1"]["kappa"],
                "agreement_prod_run2": block["pairs"]["prod_vs_run2"]["agreement"],
                "kappa_prod_run2": block["pairs"]["prod_vs_run2"]["kappa"],
                "agreement_run1_run2": block["pairs"]["run1_vs_run2"]["agreement"],
                "kappa_run1_run2": block["pairs"]["run1_vs_run2"]["kappa"]})
    for cut, body in results["confidence"].items():
        for level, block in body.items():
            if block.get("n"):
                rows.append({"section": "confidence", "layer": cut,
                             "field": level,
                             "label": f"complete M1-M7 set, confidence={level}",
                             "n": block["n"], "stable_3runs": block.get("stable"),
                             "stable_share": block.get("share")})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Repeatability of Gates 3 and 4 across the production run and two re-runs")
    parser.add_argument("--deposit", type=Path, required=True,
                        help="deposit package directory (validation.csv + main table)")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()

    source = find_validation(args.deposit)
    retest, layer = load_retest(source)
    if not retest:
        raise SystemExit(f"no `retest` block in {source}")
    production, criteria = load_production(args.deposit, set(layer))
    missing = set(layer) - set(production.index)
    if missing:
        raise SystemExit(f"{len(missing)} re-tested maps are absent from the main table")

    labels = Labels(production, criteria, retest)
    codings = load_codings(source)
    confidence = production["classification_confidence"].to_dict()

    results = {
        "description": "Test-retest of the StatMapCorpus labels: the production run "
                       "against two fresh runs (see ../ANALYSIS_PLANS.md)",
        "source": str(source),
        "sample": {"n": len(layer),
                   **pd.Series(list(layer.values())).value_counts().to_dict()},
        "gate4": gate4_section(layer, labels),
        "label_repeatability": label_repeatability(layer, labels),
        "gate3": gate3_section(layer, labels),
        "confidence": confidence_section(layer, labels, confidence),
    }
    gold, pairs = gold_section(layer, labels, codings)
    results["gold_standard"] = gold

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "retest.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    table = build_csv(results)
    table.to_csv(args.out / "retest.csv", index=False)
    if not pairs.empty:
        pairs.to_csv(args.out / "unstable_vs_gold.csv", index=False)

    print(f"\nwritten: {args.out / 'retest.json'} and {args.out / 'retest.csv'} "
          f"({len(table)} rows)")
    for cut in ("random", "validation"):
        g4 = results["gate4"][cut]
        print(f"\n[{cut:11}] n = {g4['n']}")
        print(f"  complete M1-M7 set stable in all three runs: "
              f"{g4['stable_methods7']['stable']}/{g4['stable_methods7']['n']} "
              f"= {g4['stable_methods7']['share']:.1%}")
        print(f"  ... with the legend as well:                  "
              f"{g4['stable_methods7_plus_legend']['stable']}"
              f"/{g4['stable_methods7_plus_legend']['n']} "
              f"= {g4['stable_methods7_plus_legend']['share']:.1%}")
        g3 = results["gate3"][cut]
        print(f"  would fail gate 3 in at least one run:        "
              f"{g3['fail_in_either_run']['n']}/{g3['n']} "
              f"= {g3['fail_in_either_run']['share']:.1%}")
        textless = g3["textless_and_language_other_than_en"]
        print(f"  text-free and language other than 'en':       "
              f"{textless['in_at_least_one_run']['n']}"
              f"/{textless['production_textless_n']} of the maps the production "
              f"run called text-free")

    print("\nlabel repeatability among production positives (validation layer)")
    for field, block in results["label_repeatability"]["validation"].items():
        if block["n_positive"]:
            print(f"  {block['label']:<20} {block['stable_all3']:>3}"
                  f"/{block['n_positive']:<3} = {block['share']:.1%}")

    gold = results["gold_standard"]
    if "table_2x2" in gold:
        t = gold["table_2x2"]
        print(f"\nunstable labels vs the gold standard (variant C, "
              f"{gold['n_pairs']} map-field pairs)")
        print(f"  error among stable labels:   {gold['error_among_stable']:.1%} "
              f"(n = {t['stable_right'] + t['stable_wrong']})")
        print(f"  error among unstable labels: {gold['error_among_unstable']:.1%} "
              f"(n = {t['unstable_right'] + t['unstable_wrong']})")
        if gold.get("odds_ratio_error_if_unstable"):
            print(f"  odds ratio {gold['odds_ratio_error_if_unstable']:.1f}, "
                  f"Fisher p = {gold['fisher_p_two_sided']:.2g}")


if __name__ == "__main__":
    main()
