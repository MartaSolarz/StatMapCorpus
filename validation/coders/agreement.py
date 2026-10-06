import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
CODEBOOK = HERE.parent / "codebook" / "gates.yaml"

MISSING = "cannot_assess"
BOOTSTRAP_RESAMPLES = 2000
SEED = 42
LOW, HIGH = 2.5, 97.5
DISAGREEMENT_ALERT = 0.10

CODERS = ("coder1", "coder2")


def find_validation(deposit: Path) -> Path:
    for candidate in (deposit / "validation.csv", deposit / "data" / "validation.csv"):
        if candidate.exists():
            return candidate
    raise SystemExit(f"validation.csv not found under {deposit}")


def find_main_table(deposit: Path) -> Path:
    for directory in (deposit, deposit / "data"):
        found = sorted(directory.glob("statmapcorpus_dataset*.parquet")) \
            if directory.is_dir() else []
        if found:
            return found[0]
    raise SystemExit(f"main table statmapcorpus_dataset.parquet not found under {deposit}")


def load_codings(path: Path) -> dict:
    codings = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["block"] != "coders":
                continue
            codings[(row["source"], row["uid"], row["field"])] = row["value"]
    return codings


def load_weights(path: Path) -> dict:
    weights = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["block"] != "coders" or not row["inclusion_probability"]:
                continue
            weights[row["uid"]] = float(row["inclusion_probability"])
    return weights


def load_labels(path: Path = CODEBOOK) -> tuple[dict, dict]:
    if not path.exists():
        return {}, {}
    book = yaml.safe_load(path.read_text(encoding="utf-8"))
    labels, sections = {}, {}
    for section in ("inclusion", "methods", "attributes"):
        for item in book.get(section) or []:
            code = item["code"]
            name = item.get("name") or item.get("label") or code
            labels[code] = f"{item.get('label', code)} · {name}"
            sections[code] = section
    return labels, sections


def field_order(codings: dict, sections: dict) -> list:
    rank = {"inclusion": 0, "methods": 1, "attributes": 2}
    fields = {field for _, _, field in codings}
    return sorted(fields, key=lambda f: (rank.get(sections.get(f, ""), 3), f))


def cohen_kappa(pairs: list) -> float:
    if not pairs:
        return float("nan")
    categories = sorted({value for pair in pairs for value in pair})
    index = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)))
    for first, second in pairs:
        matrix[index[first], index[second]] += 1
    total = matrix.sum()
    p_o = float(np.trace(matrix) / total)
    rows = matrix.sum(axis=1) / total
    cols = matrix.sum(axis=0) / total
    p_e = float((rows * cols).sum())
    if np.isclose(p_e, 1.0):
        return float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def bootstrap_ci(pairs: list, resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = SEED) -> tuple:
    if len(pairs) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        picked = rng.integers(0, len(pairs), len(pairs))
        value = cohen_kappa([pairs[i] for i in picked])
        if not np.isnan(value):
            values.append(value)
    if len(values) < resamples // 10:
        return float("nan"), float("nan")
    return float(np.percentile(values, LOW)), float(np.percentile(values, HIGH))


def prevalence_bias(pairs: list) -> tuple:
    categories = sorted({value for pair in pairs for value in pair})
    if len(categories) != 2:
        return float("nan"), float("nan")
    first, second = categories
    total = len(pairs)
    a = sum(1 for x, y in pairs if x == first and y == first)
    d = sum(1 for x, y in pairs if x == second and y == second)
    b = sum(1 for x, y in pairs if x == first and y == second)
    c = sum(1 for x, y in pairs if x == second and y == first)
    return abs(a - d) / total, abs(b - c) / total


def analyse_field(field: str, coders: tuple, codings: dict, uids: list) -> dict:
    first, second = coders
    pairs, dropped, disagreements = [], [], []
    for uid in uids:
        left = codings.get((first, uid, field))
        right = codings.get((second, uid, field))
        if left is None or right is None:
            continue
        if left == MISSING or right == MISSING:
            dropped.append(uid)
            continue
        pairs.append((left, right))
        if left != right:
            disagreements.append((uid, left, right))

    n = len(pairs)
    agreement = sum(1 for x, y in pairs if x == y) / n if n else float("nan")
    kappa = cohen_kappa(pairs)
    low, high = bootstrap_ci(pairs)
    prevalence, bias = prevalence_bias(pairs)
    return {
        "field": field,
        "n": n,
        "dropped": len(dropped),
        "dropped_uids": dropped,
        "agreement": agreement,
        "kappa": kappa,
        "ci_low": low,
        "ci_high": high,
        "prevalence_index": prevalence,
        "bias_index": bias,
        "disagreements": disagreements,
        "n_disagreements": len(disagreements),
        "share_disagreements": len(disagreements) / n if n else float("nan"),
        "marginals": {
            first: dict(Counter(x for x, _ in pairs)),
            second: dict(Counter(y for _, y in pairs)),
        },
        "confusion": {f"{x}->{y}": count for (x, y), count
                      in sorted(Counter(pairs).items())},
    }


def _fmt(value, digits=3, width=6):
    return f"{value:>{width}.{digits}f}" if value == value else f"{'-':>{width}}"


def print_table(results: list, labels: dict, coders: tuple) -> None:
    print(f"\nCoder agreement: {coders[0]} <-> {coders[1]}   (gates mode, Cohen's kappa)")
    print("=" * 110)
    print(f"{'field':<36}{'n':>5}{'drop':>6}{'agree':>8}{'kappa':>8}"
          f"{'95% CI':>18}{'PI':>7}{'BI':>7}{'disag.':>8}")
    print("-" * 110)
    for row in results:
        name = labels.get(row["field"], row["field"])
        ci = (f"[{row['ci_low']:.2f}; {row['ci_high']:.2f}]"
              if row["ci_low"] == row["ci_low"] else "-")
        flag = "  !" if row["share_disagreements"] > DISAGREEMENT_ALERT else ""
        print(f"{name:<36}{row['n']:>5}{row['dropped']:>6}"
              f"{_fmt(row['agreement'], 3, 8)}{_fmt(row['kappa'], 3, 8)}"
              f"{ci:>18}{_fmt(row['prevalence_index'], 2, 7)}"
              f"{_fmt(row['bias_index'], 2, 7)}"
              f"{row['n_disagreements']:>6}{flag}")
    print("-" * 110)
    total = sum(row["n_disagreements"] for row in results)
    print(f"disagreements in total: {total} "
          f"out of {sum(row['n'] for row in results)} comparisons")
    print("! = above 10% disagreement, a signal that the field definition is imprecise")


def print_disagreements(results: list, labels: dict, coders: tuple) -> None:
    print("\nDisagreements per field")
    print("=" * 110)
    for row in results:
        if not row["disagreements"]:
            continue
        print(f"\n{labels.get(row['field'], row['field'])}"
              f"  ({row['n_disagreements']}/{row['n']}, "
              f"{row['share_disagreements']:.1%})")
        directions = Counter((left, right) for _, left, right in row["disagreements"])
        for (left, right), count in directions.most_common():
            print(f"    {coders[0]}={left:<14} {coders[1]}={right:<14} {count:>3}x")


SUMMARY_COLUMNS = ["field", "label", "n", "dropped", "agreement", "kappa",
                   "ci_low", "ci_high", "prevalence_index", "bias_index",
                   "n_disagreements", "share_disagreements"]


def write_outputs(results: list, labels: dict, coders: tuple, out: Path,
                  source: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    summary = out / "agreement.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow({key: (labels.get(row["field"], row["field"])
                                   if key == "label" else row[key])
                             for key in SUMMARY_COLUMNS})

    disagreement_file = out / "disagreements.csv"
    with disagreement_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "label", "uid", coders[0], coders[1]])
        for row in results:
            for uid, left, right in row["disagreements"]:
                writer.writerow([row["field"], labels.get(row["field"], row["field"]),
                                 uid, left, right])

    detail = out / "agreement.json"
    detail.write_text(json.dumps({
        "source": str(source),
        "coders": list(coders),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": SEED},
        "fields": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nwritten:\n  {summary}\n  {disagreement_file}\n  {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agreement between the two coders (block coders of validation.csv)")
    parser.add_argument("--deposit", type=Path, required=True,
                        help="deposit package directory (holds validation.csv)")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    parser.add_argument("--show-disagreements", action="store_true",
                        help="print the direction of the disagreements per field")
    args = parser.parse_args()

    source = find_validation(args.deposit)
    codings = load_codings(source)
    if not codings:
        raise SystemExit(f"no `coders` block in {source}")

    present = sorted({coder for coder, _, _ in codings})
    if tuple(present) != CODERS:
        raise SystemExit(f"expected sources {CODERS}, found {tuple(present)}")

    by_coder = defaultdict(set)
    for coder, uid, _ in codings:
        by_coder[coder].add(uid)
    shared = sorted(by_coder[CODERS[0]] & by_coder[CODERS[1]])
    if not shared:
        raise SystemExit("the coders do not share a single map")

    only_first = len(by_coder[CODERS[0]] - by_coder[CODERS[1]])
    only_second = len(by_coder[CODERS[1]] - by_coder[CODERS[0]])
    print(f"maps in common: {len(shared)}"
          + (f"  (only {CODERS[0]}: {only_first}, only {CODERS[1]}: {only_second})"
             if only_first or only_second else ""))

    labels, sections = load_labels()
    results = [analyse_field(field, CODERS, codings, shared)
               for field in field_order(codings, sections)]

    print_table(results, labels, CODERS)
    if args.show_disagreements:
        print_disagreements(results, labels, CODERS)
    write_outputs(results, labels, CODERS, args.out, source)


if __name__ == "__main__":
    main()
