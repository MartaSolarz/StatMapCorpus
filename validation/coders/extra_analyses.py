import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from agreement import (CODERS, MISSING, find_main_table, find_validation,
                       load_codings, load_labels, load_weights)
from precision import (METHOD_COLUMN, VARIANTS, build_gold, kish_n_eff,
                       model_values, wilson)

HERE = Path(__file__).resolve().parent

METHODS = tuple(METHOD_COLUMN)
CRITERIA = ("i2", "i3", "i4")
WEAK_TYPES = ("m3", "m4", "m5", "m7")
CONFIDENCE_LEVELS = ("high", "medium", "low")


def weighted_share(uids: list, hit: list, weights: dict) -> dict:
    if not uids:
        return {"n": 0, "n_eff": 0.0, "share": float("nan"),
                "share_raw": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_hits": 0}
    w = np.array([1.0 / weights[uid] for uid in uids])
    h = np.array(hit, dtype=float)
    share = float((w * h).sum() / w.sum())
    n_eff = kish_n_eff(w)
    low, high = wilson(share, n_eff)
    return {"n": len(uids), "n_eff": n_eff, "share": share,
            "share_raw": float(h.mean()), "ci_low": low, "ci_high": high,
            "n_hits": int(h.sum())}


def complete_type_set(codings: dict, uids: list, model: dict,
                      weights: dict) -> dict:
    out = {}
    for variant in VARIANTS:
        gold = {field: build_gold(field, codings, uids, variant) for field in METHODS}
        resolved = [u for u in uids if all(u in gold[f] for f in METHODS)]
        hit = [all((gold[f][u] == "yes") == (model[f][u] == "yes") for f in METHODS)
               for u in resolved]
        out[variant] = weighted_share(resolved, hit, weights)
    return out


def joint_criteria(codings: dict, uids: list, weights: dict) -> dict:
    out = {}
    for variant in VARIANTS:
        gold = {field: build_gold(field, codings, uids, variant) for field in CRITERIA}
        resolved = [u for u in uids if all(u in gold[f] for f in CRITERIA)]
        hit = [all(gold[f][u] == "yes" for f in CRITERIA) for u in resolved]
        out[variant] = weighted_share(resolved, hit, weights)
    return out


def coder_independence(codings: dict, uids: list, model: dict) -> dict:
    cells = []
    for uid in uids:
        for field in METHODS:
            left = codings.get((CODERS[0], uid, field))
            right = codings.get((CODERS[1], uid, field))
            if left is None or right is None or MISSING in (left, right):
                continue
            model_yes = model[field][uid] == "yes"
            cells.append(((left == "yes") == model_yes,
                          (right == "yes") == model_yes))

    n = len(cells)
    first = sum(1 for a, _ in cells if a)
    second = sum(1 for _, b in cells if b)
    only_first = sum(1 for a, b in cells if a and not b)
    only_second = sum(1 for a, b in cells if b and not a)
    discordant = only_first + only_second
    p = (float(binomtest(only_first, discordant, 0.5).pvalue)
         if discordant else float("nan"))
    return {
        "n_cells": n,
        "agreement_coder1": first / n if n else float("nan"),
        "agreement_coder2": second / n if n else float("nan"),
        "n_agree_coder1": first,
        "n_agree_coder2": second,
        "discordant_only_coder1_right": only_first,
        "discordant_only_coder2_right": only_second,
        "mcnemar_exact_p_two_sided": p,
        "caveat": "cells within one map are not independent; this is a diagnostic "
                  "of asymmetry between the coders, not a test about the corpus",
    }


def by_confidence(codings: dict, uids: list, model: dict, confidence: dict,
                  variant: str = "C") -> dict:
    gold = {field: build_gold(field, codings, uids, variant) for field in METHODS}
    resolved = [u for u in uids if all(u in gold[f] for f in METHODS)]
    out = {"variant": variant, "n_resolved": len(resolved), "levels": {}}
    for level in CONFIDENCE_LEVELS:
        selected = [u for u in resolved if confidence.get(u) == level]
        if not selected:
            out["levels"][level] = {"n": 0}
            continue
        hits = sum(1 for u in selected
                   if all((gold[f][u] == "yes") == (model[f][u] == "yes")
                          for f in METHODS))
        low, high = wilson(hits / len(selected), len(selected))
        out["levels"][level] = {"n": len(selected), "n_hits": hits,
                                "share": hits / len(selected),
                                "ci_low": low, "ci_high": high}
    return out


def error_patterns(codings: dict, uids: list, model: dict,
                   variant: str = "C") -> dict:
    gold = {field: build_gold(field, codings, uids, variant) for field in METHODS}
    out = {"variant": variant, "types": {}}
    for field in WEAK_TYPES:
        false_positives = [u for u in uids
                           if model[field][u] == "yes"
                           and u in gold[field] and gold[field][u] == "no"]
        assigned_any, sole, none = Counter(), Counter(), 0
        for uid in false_positives:
            given = [f for f in METHODS if gold.get(f, {}).get(uid) == "yes"]
            for f in given:
                assigned_any[f] += 1
            if len(given) == 1:
                sole[given[0]] += 1
            elif not given:
                none += 1
        out["types"][field] = {
            "n_false_positives": len(false_positives),
            "assigned_any": dict(sorted(assigned_any.items())),
            "sole_type": dict(sorted(sole.items())),
            "no_type_at_all": none,
            "uids": false_positives,
        }
    return out


def _p(value, width=7, digits=3):
    return f"{value:>{width}.{digits}f}" if value == value else f"{'-':>{width}}"


def print_report(results: dict, labels: dict) -> None:
    print("\n1. Complete set of seven types matches the coders")
    print("=" * 96)
    print(f"{'variant':<10}{'n':>5}{'n_eff':>8}{'weighted':>11}{'raw':>9}"
          f"{'95% CI':>20}")
    print("-" * 96)
    for variant in VARIANTS:
        body = results["complete_type_set"][variant]
        ci = (f"[{body['ci_low']:.2f}; {body['ci_high']:.2f}]"
              if body["ci_low"] == body["ci_low"] else "-")
        print(f"{variant:<10}{body['n']:>5}{body['n_eff']:>8.1f}"
              f"{_p(body['share'], 11)}{_p(body['share_raw'], 9)}{ci:>20}")

    print("\n2. The three statistical-map criteria jointly (i2 AND i3 AND i4 = yes)")
    print("=" * 96)
    print(f"{'variant':<10}{'n':>5}{'n_eff':>8}{'weighted':>11}{'raw':>9}"
          f"{'95% CI':>20}")
    print("-" * 96)
    for variant in VARIANTS:
        body = results["joint_criteria"][variant]
        ci = (f"[{body['ci_low']:.2f}; {body['ci_high']:.2f}]"
              if body["ci_low"] == body["ci_low"] else "-")
        print(f"{variant:<10}{body['n']:>5}{body['n_eff']:>8.1f}"
              f"{_p(body['share'], 11)}{_p(body['share_raw'], 9)}{ci:>20}")

    ind = results["coder_independence"]
    print(f"\n3. Independence check: model vs each coder on all m1-m7 cells "
          f"(n = {ind['n_cells']:,})")
    print("=" * 96)
    print(f"  {CODERS[0]} agrees with the model: {ind['n_agree_coder1']:>5} "
          f"= {ind['agreement_coder1']:.1%}")
    print(f"  {CODERS[1]} agrees with the model: {ind['n_agree_coder2']:>5} "
          f"= {ind['agreement_coder2']:.1%}")
    print(f"  discordant pairs: {ind['discordant_only_coder1_right']} "
          f"({CODERS[0]} right alone) vs {ind['discordant_only_coder2_right']} "
          f"({CODERS[1]} right alone)")
    print(f"  exact McNemar, two-sided: p = {ind['mcnemar_exact_p_two_sided']:.4f}")
    print(f"  caveat: {ind['caveat']}")

    conf = results["by_confidence"]
    print(f"\n4. Complete type set by the model's declared confidence "
          f"(variant {conf['variant']}, n = {conf['n_resolved']})")
    print("=" * 96)
    print(f"{'confidence':<12}{'n':>5}{'matched':>9}{'share':>9}{'95% CI':>20}")
    print("-" * 96)
    for level in CONFIDENCE_LEVELS:
        body = conf["levels"][level]
        if not body["n"]:
            print(f"{level:<12}{0:>5}        -        -                   -")
            continue
        ci = f"[{body['ci_low']:.2f}; {body['ci_high']:.2f}]"
        print(f"{level:<12}{body['n']:>5}{body['n_hits']:>9}"
              f"{_p(body['share'], 9)}{ci:>20}")

    pat = results["error_patterns"]
    print(f"\n5. Error patterns: what the coders assigned to the false positives "
          f"(variant {pat['variant']})")
    print("=" * 96)
    for field, body in pat["types"].items():
        n = body["n_false_positives"]
        print(f"\n  false {field.upper()} ({labels.get(field, field)}): {n} maps")
        if not n:
            continue
        for other, count in body["assigned_any"].items():
            sole = body["sole_type"].get(other, 0)
            print(f"      coders assigned {other.upper():<4} "
                  f"{count}/{n} maps   (as the only type: {sole}/{n})")
        if body["no_type_at_all"]:
            print(f"      coders assigned no type at all: "
                  f"{body['no_type_at_all']}/{n}")


def write_outputs(results: dict, out: Path, source: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANTS:
        for name in ("complete_type_set", "joint_criteria"):
            body = results[name][variant]
            rows.append({"analysis": name, "variant": variant, "key": "",
                         "n": body["n"], "n_eff": round(body["n_eff"], 3),
                         "value": body["share"], "value_raw": body["share_raw"],
                         "ci_low": body["ci_low"], "ci_high": body["ci_high"]})
    ind = results["coder_independence"]
    for key in ("agreement_coder1", "agreement_coder2"):
        rows.append({"analysis": "coder_independence", "variant": "",
                     "key": key, "n": ind["n_cells"], "n_eff": "",
                     "value": ind[key], "value_raw": ind[key],
                     "ci_low": "", "ci_high": ""})
    rows.append({"analysis": "coder_independence", "variant": "",
                 "key": "mcnemar_exact_p_two_sided", "n": ind["n_cells"],
                 "n_eff": "", "value": ind["mcnemar_exact_p_two_sided"],
                 "value_raw": "", "ci_low": "", "ci_high": ""})
    conf = results["by_confidence"]
    for level, body in conf["levels"].items():
        if not body["n"]:
            continue
        rows.append({"analysis": "by_confidence", "variant": conf["variant"],
                     "key": level, "n": body["n"], "n_eff": "",
                     "value": body["share"], "value_raw": body["share"],
                     "ci_low": body["ci_low"], "ci_high": body["ci_high"]})
    pat = results["error_patterns"]
    for field, body in pat["types"].items():
        for other, count in body["assigned_any"].items():
            rows.append({"analysis": "error_patterns", "variant": pat["variant"],
                         "key": f"false_{field}_coders_said_{other}",
                         "n": body["n_false_positives"], "n_eff": "",
                         "value": count / body["n_false_positives"],
                         "value_raw": count, "ci_low": "", "ci_high": ""})

    table = out / "extra_analyses.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "analysis", "variant", "key", "n", "n_eff", "value", "value_raw",
            "ci_low", "ci_high"])
        writer.writeheader()
        writer.writerows(rows)

    detail = out / "extra_analyses.json"
    detail.write_text(json.dumps({"source": str(source), "coders": list(CODERS),
                                  **results}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nwritten:\n  {table}\n  {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete type sets, joint criteria, coder independence, confidence and error patterns")
    parser.add_argument("--deposit", type=Path, required=True,
                        help="deposit package directory (validation.csv + main table)")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()

    source = find_validation(args.deposit)
    codings = load_codings(source)
    weights = load_weights(source)
    if not codings:
        raise SystemExit(f"no `coders` block in {source}")

    uids = sorted({uid for coder, uid, _ in codings if coder == CODERS[0]}
                  & {uid for coder, uid, _ in codings if coder == CODERS[1]})

    frame = pd.read_parquet(find_main_table(args.deposit))
    frame = frame[frame["uid"].isin(uids)].reset_index(drop=True)
    if set(uids) - set(frame["uid"]):
        raise SystemExit("some sampled maps are absent from the main table")

    model = {field: model_values(field, frame) for field in METHODS}
    confidence = dict(zip(frame["uid"], frame["classification_confidence"]))

    print(f"maps in the sample: {len(uids)}")
    results = {
        "complete_type_set": complete_type_set(codings, uids, model, weights),
        "joint_criteria": joint_criteria(codings, uids, weights),
        "coder_independence": coder_independence(codings, uids, model),
        "by_confidence": by_confidence(codings, uids, model, confidence),
        "error_patterns": error_patterns(codings, uids, model),
    }

    labels, _ = load_labels()
    print_report(results, labels)
    write_outputs(results, args.out, source)


if __name__ == "__main__":
    main()
