import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "coders"))

from agreement import find_main_table, find_validation

Z = 1.959963985
DECOY_THRESHOLD = 0.70
DECOY = "DECOY"
REJECTION_STRATA = ["G2_not_a_map", "G2_not_dominant", "G3_content",
                    "G3_language", "G4_no_type"]
STRATUM_DESCRIPTION = {
    "G2_not_a_map": "gate 2 - rejected as not a map",
    "G2_not_dominant": "gate 2 - rejected as a map that is not dominant",
    "G3_content": "gate 3 - rejected on content",
    "G3_language": "gate 3 - rejected on language alone",
    "G4_no_type": "gate 4 - passed, but with none of the types M1-M7",
    DECOY: "maps accepted into the corpus (control)",
}


def wilson(p, n):
    if n <= 0 or p is None or not 0.0 <= p <= 1.0:
        return None, None
    d = 1.0 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z / d * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def share(numerator, denominator, population=None):
    if denominator == 0:
        return {"numerator": int(numerator), "denominator": 0,
                "share": None, "ci95": [None, None]}
    p = numerator / denominator
    low, high = wilson(p, denominator)
    out = {"numerator": int(numerator), "denominator": int(denominator),
           "share": p, "ci95": [low, high]}
    if population:
        out["population_n"] = int(population)
        out["estimate_in_population"] = round(p * population)
        out["estimate_ci95"] = [round(low * population), round(high * population)]
    return out


def load_audit(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["block"] != "audit" or row["field"] != "meets_all_criteria":
                continue
            rows.append({
                "uid": row["uid"],
                "stratum": row["stratum"],
                "answer": row["value"],
                "population_n": int(row["population_n"]) if row["population_n"] else None,
                "response_time_s": (float(row["response_time_s"])
                                    if row["response_time_s"] else float("nan")),
            })
    return pd.DataFrame(rows)


def check_against_corpus(frame: pd.DataFrame, deposit: Path) -> dict:
    corpus = set(pd.read_parquet(find_main_table(deposit), columns=["uid"])["uid"])
    decoys = set(frame.loc[frame.stratum == DECOY, "uid"])
    rejected = set(frame.loc[frame.stratum != DECOY, "uid"])
    return {
        "decoys_in_corpus": len(decoys & corpus),
        "decoys_total": len(decoys),
        "rejected_found_in_corpus": sorted(rejected & corpus),
    }


def main():
    parser = argparse.ArgumentParser(description="Wrongful-rejection rates of the cascade from the blind audit (block audit of validation.csv)")
    parser.add_argument("--deposit", type=Path, required=True,
                        help="deposit package directory (validation.csv + main table)")
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()

    source = find_validation(args.deposit)
    frame = load_audit(source)
    if frame.empty:
        raise SystemExit(f"no `audit` block in {source}")

    integrity = check_against_corpus(frame, args.deposit)
    if integrity["rejected_found_in_corpus"]:
        raise SystemExit(
            f"{len(integrity['rejected_found_in_corpus'])} maps marked as rejected "
            "are present in the corpus; the strata do not hold")
    if integrity["decoys_in_corpus"] != integrity["decoys_total"]:
        raise SystemExit("some decoys are absent from the corpus; the control is void")

    assessed = frame[frame.answer.isin(["yes", "no", "cannot_assess"])]

    results = {
        "description": "Blind audit of the cascade's rejections, one question per map. "
                       "One coder.",
        "source": str(source),
        "integrity_check": integrity,
        "completeness": {
            "maps_in_sample": len(frame),
            "assessed": len(assessed),
            "median_time_s": (round(float(assessed.response_time_s.median()), 1)
                              if len(assessed) else None),
            "total_time_min": (round(float(assessed.response_time_s.sum()) / 60, 1)
                               if len(assessed) else None),
        },
        "strata": {},
    }
    unassessed = sorted(set(frame.uid) - set(assessed.uid))
    if unassessed:
        results["completeness"]["unassessed_uids"] = unassessed
        results["completeness"]["note"] = (
            f"{len(unassessed)} maps in the sample have no verdict -- the numbers "
            "cover the assessed ones only.")

    for stratum in REJECTION_STRATA:
        subset = assessed[assessed.stratum == stratum]
        if subset.empty:
            results["strata"][stratum] = {
                "description": STRATUM_DESCRIPTION[stratum], "n_assessed": 0}
            continue
        population = int(subset.population_n.iloc[0])
        decided = subset[subset.answer != "cannot_assess"]
        wrongful = int((decided.answer == "yes").sum())
        results["strata"][stratum] = {
            "description": STRATUM_DESCRIPTION[stratum],
            "n_assessed": len(subset),
            "n_dropped_cannot_assess": len(subset) - len(decided),
            "wrongful_rejection": share(wrongful, len(decided), population),
            "answer_distribution": subset.answer.value_counts().to_dict(),
            "median_time_s": round(float(subset.response_time_s.median()), 1),
        }

    decoys = assessed[assessed.stratum == DECOY]
    if not decoys.empty:
        decided = decoys[decoys.answer != "cannot_assess"]
        confirmed = int((decided.answer == "yes").sum())
        p = share(confirmed, len(decided))
        results["decoys"] = {
            "description": STRATUM_DESCRIPTION[DECOY],
            "n_assessed": len(decoys),
            "n_dropped_cannot_assess": len(decoys) - len(decided),
            "confirmed_as_meeting_the_criteria": p,
            "answer_distribution": decoys.answer.value_counts().to_dict(),
            "median_time_s": round(float(decoys.response_time_s.median()), 1),
            "interpretation": (
                None if p["share"] is None else
                "the coding is not stricter than the corpus criteria; the wrongful-rejection rates read directly"
                if p["share"] >= DECOY_THRESHOLD else
                "THE DECOY THRESHOLD IS NOT MET -- the coder applies the criteria more "
                "strictly than the model; the wrongful-rejection rates should be "
                "treated as overstated"),
        }

    total = low = high = 0
    for stratum in REJECTION_STRATA:
        body = results["strata"].get(stratum, {}).get("wrongful_rejection", {})
        if body.get("estimate_in_population") is not None:
            total += body["estimate_in_population"]
            low += body["estimate_ci95"][0]
            high += body["estimate_ci95"][1]
    results["summary"] = {
        "estimated_wrongfully_rejected_maps": total,
        "interval": [low, high],
        "caveat": "Extrapolated from samples of 15-60 maps onto strata of up to 26 "
                  "thousand; the interval is the sum of the Wilson bounds across "
                  "strata, not a joint interval.",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    rows = []
    for stratum in REJECTION_STRATA:
        body = results["strata"].get(stratum, {})
        if not body.get("n_assessed"):
            continue
        e = body["wrongful_rejection"]
        rows.append({
            "stratum": stratum, "description": body["description"],
            "population_n": e.get("population_n"),
            "n_assessed": body["n_assessed"], "n_decided": e["denominator"],
            "n_cannot_assess": body["n_dropped_cannot_assess"],
            "wrongful_rejections": e["numerator"], "share": e["share"],
            "ci_low": e["ci95"][0], "ci_high": e["ci95"][1],
            "estimate_in_population": e.get("estimate_in_population"),
            "estimate_low": (e.get("estimate_ci95") or [None, None])[0],
            "estimate_high": (e.get("estimate_ci95") or [None, None])[1],
        })
    if "decoys" in results:
        d = results["decoys"]
        p = d["confirmed_as_meeting_the_criteria"]
        rows.append({"stratum": DECOY, "description": d["description"],
                     "population_n": None, "n_assessed": d["n_assessed"],
                     "n_decided": p["denominator"],
                     "n_cannot_assess": d["n_dropped_cannot_assess"],
                     "wrongful_rejections": None, "share": p["share"],
                     "ci_low": p["ci95"][0], "ci_high": p["ci95"][1]})
    pd.DataFrame(rows).to_csv(args.out / "audit.csv", index=False)
    frame[["uid", "stratum", "answer", "response_time_s"]].to_csv(
        args.out / "audit_per_map.csv", index=False)

    print(f"assessed {len(assessed)} / {len(frame)} maps"
          + (f"   MISSING {len(unassessed)}" if unassessed else ""))
    print(f"integrity: {integrity['decoys_in_corpus']}/{integrity['decoys_total']} "
          "decoys are corpus records, no rejected map is")
    print(f"median time per map: {results['completeness']['median_time_s']} s\n")
    print(f"{'stratum':18}{'asd':>5}{'dec':>5}{'wrong':>8}{'share':>10}{'CI95':>16}"
          f"{'in the population':>28}")
    for stratum in REJECTION_STRATA:
        body = results["strata"].get(stratum, {})
        if not body.get("n_assessed"):
            continue
        e = body["wrongful_rejection"]
        if e["share"] is None:
            continue
        print(f"{stratum:18}{body['n_assessed']:>5}{e['denominator']:>5}"
              f"{e['numerator']:>8}{e['share'] * 100:>9.1f}%"
              f"   [{e['ci95'][0] * 100:>4.1f}, {e['ci95'][1] * 100:>5.1f}]"
              f"   {e['estimate_in_population']:>7} of {e['population_n']:<6}"
              f" [{e['estimate_ci95'][0]}, {e['estimate_ci95'][1]}]")
    if "decoys" in results:
        p = results["decoys"]["confirmed_as_meeting_the_criteria"]
        print(f"\nDECOYS: {p['numerator']}/{p['denominator']} "
              f"= {p['share'] * 100:.1f}% \"yes\"")
        print(f"  -> {results['decoys']['interpretation']}")
    s = results["summary"]
    print(f"\nestimated number of maps rejected wrongly: "
          f"{s['estimated_wrongfully_rejected_maps']} "
          f"(interval {s['interval'][0]}-{s['interval'][1]})")
    print(f"\nwritten: {args.out / 'audit.json'}, {args.out / 'audit.csv'}, "
          f"{args.out / 'audit_per_map.csv'}")


if __name__ == "__main__":
    main()
