import argparse
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

CODER_ORDER = ("marta", "koder1")

METHOD_COLUMN = {
    "m1": "method_choropleth",
    "m2": "method_diagrams",
    "m3": "method_isolines",
    "m4": "method_dot_density",
    "m5": "method_heat_map",
    "m6": "method_cartogram",
    "m7": "method_flow_map",
}

RETEST_RENAME = {
    "has_choropleth": "m1", "has_diagrams": "m2", "has_isolines": "m3",
    "has_dot_density": "m4", "has_heat_map": "m5", "has_cartogram": "m6",
    "has_flow_map": "m7",
}
RETEST_GATE3 = ("is_statistical_map", "has_admin_units",
                "has_quantitative_data", "has_text", "map_language")

LEGEND_MAP = {True: "classed", False: "continuous", None: "other"}

AUDIT_STRATUM = {
    "G2_niemapa": "G2_not_a_map",
    "G2_niedominujaca": "G2_not_dominant",
    "G3_tresc": "G3_content",
    "G3_jezyk": "G3_language",
    "G4_brak_typu": "G4_no_type",
    "WABIKI": "DECOY",
}

STRATUM_NAME = AUDIT_STRATUM | {name: name for name in AUDIT_STRATUM.values()}

COLUMNS = ["block", "uid", "stratum", "source", "field", "value",
           "inclusion_probability", "population_n", "response_time_s"]


def load_codings(path: Path, mode: str = "gates") -> dict:
    best: dict = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("mode") != mode:
                continue
            if str(row.get("superseded", "0")).strip() in {"1", "true", "True"}:
                continue
            key = (row["coder"], row["uid"], row["field"])
            revision = int(row.get("revision") or 0)
            if key not in best or revision > best[key][0]:
                best[key] = (revision, row["value"])
    return {key: value for key, (_, value) in best.items()}


AUDIT_KEYS = {"strata": ("strata", "warstwy"),
              "drawn": ("drawn", "wylosowano"),
              "pool": ("positives_in_corpus", "pozytywow_w_korpusie"),
              "column": ("column", "kolumna")}
CODING_KEYS = {"answer": ("answer", "odpowiedz"),
               "time_ms": ("time_ms", "czas_ms")}


def pick(mapping: dict, names: tuple):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise SystemExit(f"none of the keys {names} is present; got {sorted(mapping)}")


def inclusion_probabilities(frame: pd.DataFrame, audit: dict) -> pd.Series:
    complement = pd.Series(1.0, index=frame.index)
    for body in pick(audit, AUDIT_KEYS["strata"]).values():
        share = pick(body, AUDIT_KEYS["drawn"]) / pick(body, AUDIT_KEYS["pool"])
        column = pick(body, AUDIT_KEYS["column"])
        if column == "has_text == false":
            member = ~frame["has_text"].astype(bool)
        else:
            member = frame[column].astype(bool)
        complement = complement * np.where(member, 1.0 - share, 1.0)
    return 1.0 - complement


def coders_rows(codings: dict, corpus: pd.DataFrame, audit: dict) -> list:
    uids = sorted({u for c, u, _ in codings if c == CODER_ORDER[0]}
                  & {u for c, u, _ in codings if c == CODER_ORDER[1]})
    frame = corpus[corpus["uid"].isin(uids)].reset_index(drop=True)
    missing = set(uids) - set(frame["uid"])
    if missing:
        raise SystemExit(f"{len(missing)} coded maps are absent from the corpus")
    pi = dict(zip(frame["uid"], inclusion_probabilities(frame, audit)))

    rows = []
    for (coder, uid, field), value in sorted(codings.items()):
        if coder not in CODER_ORDER or uid not in pi:
            continue
        rows.append({
            "block": "coders", "uid": uid, "stratum": "",
            "source": f"coder{CODER_ORDER.index(coder) + 1}",
            "field": field, "value": value,
            "inclusion_probability": f"{pi[uid]:.10g}",
            "population_n": "", "response_time_s": "",
        })
    return rows


def retest_rows(db: Path, sample: Path) -> list:
    layer = {"walidacyjna": "validation", "losowa": "random",
             "validation": "validation", "random": "random"}
    frame = pd.read_csv(sample)
    layer_col = "layer" if "layer" in frame.columns else "warstwa"
    stratum = {u: layer[w] for u, w in zip(frame["uid"], frame[layer_col])}

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    runs = pd.read_sql("SELECT uid, run, gate, status, parsed_json FROM runs", con)
    con.close()

    rows = []
    for record in runs.itertuples(index=False):
        if record.status != "done" or not record.parsed_json:
            continue
        parsed = json.loads(record.parsed_json)
        for key, value in parsed.items():
            field = RETEST_RENAME.get(key, key)
            if field not in set(RETEST_RENAME.values()) | set(RETEST_GATE3) \
                    | {"legend_is_classed"}:
                continue
            if field == "legend_is_classed":
                plain = LEGEND_MAP[None if value is None
                                   or (isinstance(value, float) and value != value)
                                   else bool(value)]
            else:
                plain = _plain(value)
            rows.append({
                "block": "retest", "uid": record.uid,
                "stratum": stratum.get(record.uid, ""), "source": record.run,
                "field": field, "value": plain,
                "inclusion_probability": "", "population_n": "",
                "response_time_s": "",
            })
    return rows


def _plain(value):
    if value is None or (isinstance(value, float) and value != value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def audit_rows(codings: Path, sample: Path) -> list:
    coded = pd.read_csv(codings, dtype=str).fillna("")
    frame = pd.read_csv(sample)
    meta = frame.set_index("uid")
    answer_col = next(c for c in CODING_KEYS["answer"] if c in coded.columns)
    time_col = next(c for c in CODING_KEYS["time_ms"] if c in coded.columns)
    stratum_col = "stratum" if "stratum" in frame.columns else "warstwa"
    population_col = ("population_n" if "population_n" in frame.columns
                      else "N_populacji")
    rows = []
    for record in coded.to_dict("records"):
        if record["uid"] not in meta.index or record[answer_col] == "":
            continue
        body = meta.loc[record["uid"]]
        seconds = pd.to_numeric(record[time_col], errors="coerce")
        rows.append({
            "block": "audit", "uid": record["uid"],
            "stratum": STRATUM_NAME[body[stratum_col]], "source": "coder1",
            "field": "meets_all_criteria", "value": record[answer_col],
            "inclusion_probability": "",
            "population_n": int(body[population_col]),
            "response_time_s": ("" if seconds != seconds
                                else f"{seconds / 1000:.3f}"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validation.csv from the working sources of the three validation exercises")
    parser.add_argument("--codings", type=Path, required=True)
    parser.add_argument("--sample-audit", type=Path, required=True,
                        help="sampling audit of the coder validation sample")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--retest-db", type=Path, required=True)
    parser.add_argument("--retest-sample", type=Path, required=True)
    parser.add_argument("--audit-codings", type=Path, required=True)
    parser.add_argument("--audit-sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("validation.csv"))
    args = parser.parse_args()

    corpus = pd.read_parquet(args.corpus)
    audit = json.loads(args.sample_audit.read_text(encoding="utf-8"))

    rows = coders_rows(load_codings(args.codings), corpus, audit)
    rows += retest_rows(args.retest_db, args.retest_sample)
    rows += audit_rows(args.audit_codings, args.audit_sample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    frame = pd.DataFrame(rows)
    print(f"wrote {args.out}  ({len(rows):,} rows)")
    for block, group in frame.groupby("block"):
        print(f"  {block:8} {len(group):>6,} rows   "
              f"{group.uid.nunique():>4} maps   "
              f"{group.field.nunique():>2} fields   "
              f"sources: {', '.join(sorted(group.source.unique()))}")


if __name__ == "__main__":
    main()
