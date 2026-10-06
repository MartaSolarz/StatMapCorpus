import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agreement import (CODERS, MISSING, cohen_kappa, find_main_table,
                       find_validation, load_codings, load_labels, load_weights)

HERE = Path(__file__).resolve().parent

Z = 1.959963985

METHOD_COLUMN = {
    "m1": "method_choropleth",
    "m2": "method_diagrams",
    "m3": "method_isolines",
    "m4": "method_dot_density",
    "m5": "method_heat_map",
    "m6": "method_cartogram",
    "m7": "method_flow_map",
}
EXTRA_COLUMNS = {"has_text": "has_text", "legend_is_classed": "legend_is_classed"}
COLUMN_OF = METHOD_COLUMN | EXTRA_COLUMNS

LEGEND_MAP = {True: "classed", False: "continuous", None: "other"}

VARIANTS = ("A", "B", "C")
REPORTED = "C"


def kish_n_eff(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum()) if len(weights) else 0.0


def wilson(p: float, n: float) -> tuple:
    if n <= 0 or not 0.0 <= p <= 1.0:
        return float("nan"), float("nan")
    denominator = 1.0 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denominator
    half = Z / denominator * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def build_gold(field: str, codings: dict, uids: list, variant: str) -> dict:
    first, second = CODERS
    gold = {}
    for uid in uids:
        left = codings.get((first, uid, field))
        right = codings.get((second, uid, field))
        if left is None or right is None or MISSING in (left, right):
            continue
        if left == right:
            gold[uid] = left
        elif variant == "A":
            gold[uid] = left
        elif variant == "B":
            gold[uid] = right
    return gold


def model_values(field: str, frame: pd.DataFrame) -> dict:
    column = COLUMN_OF[field]
    values = {}
    for uid, raw in zip(frame["uid"], frame[column]):
        if field == "legend_is_classed":
            values[uid] = LEGEND_MAP[None if raw is None or pd.isna(raw) else bool(raw)]
        else:
            values[uid] = "yes" if bool(raw) else "no"
    return values


def precision(gold: dict, model: dict, weights: dict, positive: str) -> dict:
    claimed = [uid for uid, value in model.items()
               if value == positive and uid in gold]
    if not claimed:
        return {"n_raw": 0, "n_eff": 0.0, "precision": float("nan"),
                "precision_raw": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_hits": 0}
    w = np.array([1.0 / weights[uid] for uid in claimed])
    hit = np.array([gold[uid] == positive for uid in claimed], dtype=float)
    weighted = float((w * hit).sum() / w.sum())
    n_eff = kish_n_eff(w)
    low, high = wilson(weighted, n_eff)
    return {"n_raw": len(claimed), "n_eff": n_eff, "precision": weighted,
            "precision_raw": float(hit.mean()), "ci_low": low, "ci_high": high,
            "n_hits": int(hit.sum())}


def analyse(field: str, positive: str, codings: dict, uids: list,
            model: dict, weights: dict) -> dict:
    out = {"field": field, "positive": positive, "variants": {}}
    for variant in VARIANTS:
        gold = build_gold(field, codings, uids, variant)
        pairs = [(model[uid], gold[uid]) for uid in gold]
        row = precision(gold, model, weights, positive)
        row["n_gold"] = len(gold)
        row["kappa"] = cohen_kappa(pairs)
        row["agreement"] = (sum(1 for a, b in pairs if a == b) / len(pairs)
                            if pairs else float("nan"))
        out["variants"][variant] = row
    spread = [out["variants"][v]["precision"] for v in VARIANTS]
    spread = [value for value in spread if value == value]
    out["spread"] = max(spread) - min(spread) if spread else float("nan")
    return out


def _pct(value, width=6):
    return f"{value:>{width}.3f}" if value == value else f"{'-':>{width}}"


def _name(row: dict, labels: dict) -> str:
    label = labels.get(row["field"], row["field"])
    if row["field"].startswith("m") and row["field"][1:].isdigit():
        return label
    return f"{label} = {row['positive']}"


def print_report(results: list, labels: dict) -> None:
    print("\nPrecision of the model against the gold standard (Horvitz-Thompson weighted)")
    print(f"gold variants:  A = {CODERS[0]}'s reading   B = {CODERS[1]}'s reading"
          f"   C = agreement only")
    print(f"n, n_eff and the interval are reported for variant {REPORTED}")
    print("=" * 110)
    print(f"{'field / verdict':<36}{'n':>5}{'n_eff':>7}"
          f"{'prec. A':>9}{'prec. B':>9}{'prec. C':>9}{'spread':>9}"
          f"{'95% CI (C)':>20}")
    print("-" * 110)
    for row in results:
        name = _name(row, labels)
        a, b, c = (row["variants"][v] for v in VARIANTS)
        shown = row["variants"][REPORTED]
        ci = (f"[{shown['ci_low']:.2f}; {shown['ci_high']:.2f}]"
              if shown["ci_low"] == shown["ci_low"] else "-")
        flag = "  !" if row["spread"] == row["spread"] and row["spread"] > 0.10 else ""
        print(f"{name:<36}{shown['n_raw']:>5}{shown['n_eff']:>7.1f}"
              f"{_pct(a['precision'], 9)}{_pct(b['precision'], 9)}"
              f"{_pct(c['precision'], 9)}{_pct(row['spread'], 9)}{ci:>20}{flag}")
    print("-" * 110)
    print("spread = |max - min| of precision across the gold variants;"
          " ! above 10 pp = the conclusion depends on the coder dispute")

    print("\nCohen's kappa model <-> gold standard "
          "(ceiling: the coder-to-coder kappa from agreement.py)")
    print("=" * 110)
    print(f"{'field':<36}{'kappa (A)':>11}{'kappa (B)':>11}{'kappa (C)':>11}"
          f"{'agree (C)':>11}{'n gold (C)':>12}")
    print("-" * 110)
    for field in dict.fromkeys(row["field"] for row in results):
        row = next(r for r in results if r["field"] == field)
        a, b, c = (row["variants"][v] for v in VARIANTS)
        print(f"{labels.get(field, field):<36}{_pct(a['kappa'], 11)}"
              f"{_pct(b['kappa'], 11)}{_pct(c['kappa'], 11)}"
              f"{_pct(c['agreement'], 11)}{c['n_gold']:>12}")
    print("-" * 110)


COLUMNS = ["field", "label", "positive", "variant", "n_raw", "n_eff", "n_hits",
           "precision", "precision_raw", "ci_low", "ci_high", "kappa",
           "agreement", "n_gold"]


def write_outputs(results: list, labels: dict, out: Path, source: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table = out / "precision.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in results:
            for variant, body in row["variants"].items():
                writer.writerow({"field": row["field"],
                                 "label": labels.get(row["field"], row["field"]),
                                 "positive": row["positive"], "variant": variant,
                                 **{key: body[key] for key in COLUMNS[4:]}})
    detail = out / "precision.json"
    detail.write_text(json.dumps({"source": str(source), "coders": list(CODERS),
                                  "reported_variant": REPORTED, "fields": results},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten:\n  {table}\n  {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted precision of the model labels against the coders")
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
    missing = set(uids) - set(frame["uid"])
    if missing:
        raise SystemExit(f"{len(missing)} sampled maps are absent from the main table")
    if set(uids) - set(weights):
        raise SystemExit("validation.csv has no inclusion_probability for some maps")

    pi = pd.Series([weights[u] for u in frame["uid"]])
    print(f"maps in the sample: {len(uids)}   pi: min {pi.min():.4f}, "
          f"median {pi.median():.4f}, max {pi.max():.4f}")

    plan = [(f"m{i}", "yes") for i in range(1, 8)]
    plan += [("has_text", "no"), ("has_text", "yes")]
    plan += [("legend_is_classed", value) for value in ("classed", "continuous", "other")]

    labels, _ = load_labels()
    results = [analyse(field, positive, codings, uids,
                       model_values(field, frame), weights)
               for field, positive in plan]

    print_report(results, labels)
    write_outputs(results, labels, args.out, source)


if __name__ == "__main__":
    main()
