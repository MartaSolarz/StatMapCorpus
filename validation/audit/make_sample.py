import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

SEED = 42
ORDER_SEED = 43

G2_BASE = "mt_step1_status = 'done'"
PASSED_G2 = "mt_step2_status = 'done' AND mt_is_map = 1 AND mt_is_map_dominant = 1"
CONTENT = "(mt_is_statistical_map = 1 AND mt_has_admin_units = 1 " \
          "AND mt_has_quantitative_data = 1)"

STRATA = [
    ("G2_not_a_map",     20, f"{G2_BASE} AND mt_is_map = 0"),
    ("G2_not_dominant",  20, f"{G2_BASE} AND mt_is_map = 1 AND mt_is_map_dominant = 0"),
    ("G3_content",       60, f"{PASSED_G2} AND NOT {CONTENT}"),
    ("G3_language",      20, f"{PASSED_G2} AND {CONTENT} "
                             "AND LOWER(mt_map_language) <> 'en'"),
    ("G4_no_type",       30, "mt_step3_status = 'done' AND mt_methods_count = 0"),
]
EXPECTED_N = {"G2_not_a_map": 337, "G2_not_dominant": 425, "G3_content": 25984,
              "G3_language": 17643, "G4_no_type": 97}

N_FINAL = {"G2_not_a_map": 15, "G2_not_dominant": 15, "G3_content": 60,
           "G3_language": 20, "G4_no_type": 20, "DECOY": 20}


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the rejection-audit sample from the working database")
    parser.add_argument("--db", type=Path, required=True,
                        help="working SQLite database (opened read-only)")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="main corpus table (parquet) -- the decoy pool")
    parser.add_argument("--coded", type=Path, required=True,
                        help="frozen list of the uids in the coder validation sample")
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--images", type=Path, default=None,
                        help="where to copy the images (default: <out>/images)")
    args = parser.parse_args()
    images = args.images or (args.out / "images")

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    rng = np.random.default_rng(SEED)
    rows, populations = [], {}

    for stratum, n, where in STRATA:
        pool = pd.read_sql(
            f"SELECT uid, local_path FROM candidates WHERE {where} "
            "AND local_path IS NOT NULL ORDER BY uid", con)
        populations[stratum] = len(pool)
        expected = EXPECTED_N[stratum]
        flag = "" if len(pool) == expected else f"  <-- MISMATCH, expected {expected}"
        print(f"{stratum:18} N = {len(pool):6}  taking {N_FINAL[stratum]:3} "
              f"(the first of {n} drawn with seed 42){flag}")
        index = rng.permutation(len(pool))[:n]
        for i in index[:N_FINAL[stratum]]:
            rows.append({"uid": pool.uid.iloc[i], "stratum": stratum,
                         "local_path": pool.local_path.iloc[i]})

    corpus = pd.read_parquet(args.corpus, columns=["uid", "is_duplicate_byte"])
    coded = {line.strip() for line in args.coded.read_text().splitlines()
             if line.strip()}
    pool = (corpus.loc[~corpus.is_duplicate_byte.astype(bool), "uid"]
            .loc[lambda s: ~s.isin(coded)].sort_values().to_numpy())
    populations["DECOY"] = len(pool)
    print(f"{'DECOY':18} N = {len(pool):6}  drawing {N_FINAL['DECOY']} "
          f"(the first of 40 drawn with seed 42)")
    drawn = pool[rng.permutation(len(pool))[:40]][:N_FINAL["DECOY"]]
    paths = pd.read_sql(
        f"SELECT uid, local_path FROM candidates "
        f"WHERE uid IN ({','.join('?' * len(drawn))})",
        con, params=list(drawn)).set_index("uid").local_path
    rows += [{"uid": u, "stratum": "DECOY", "local_path": paths[u]} for u in drawn]
    con.close()

    sample = pd.DataFrame(rows)
    assert sample.uid.is_unique, "the strata overlap -- the same uid in two of them"
    expected_total = sum(N_FINAL.values())
    assert len(sample) == expected_total, \
        f"expected {expected_total} maps, got {len(sample)}"

    previous_path = args.out / "sample_v1.csv"
    if previous_path.exists():
        previous = set(pd.read_csv(previous_path).uid)
        foreign = set(sample.uid) - previous
        assert not foreign, f"{len(foreign)} uids from outside the v1 sample -- " \
                            "the draw has diverged"
        print(f"\ncheck: all {len(sample)} uids come from the v1 sample "
              f"({len(previous)} maps)")

    order = np.random.default_rng(ORDER_SEED).permutation(len(sample))
    sample["position"] = np.argsort(order)
    sample = sample.sort_values("uid").reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    sample[["uid", "stratum", "position"]].assign(
        population_n=sample.stratum.map(populations)).to_csv(
            args.out / "sample.csv", index=False)

    images.mkdir(parents=True, exist_ok=True)
    missing = []
    for uid, src in zip(sample.uid, sample.local_path):
        source = Path(src)
        if not source.exists():
            missing.append(uid)
            continue
        shutil.copy2(source, images / f"{uid}{source.suffix.lower()}")
    if missing:
        raise SystemExit(f"STOP: {len(missing)} image files missing: {missing[:5]}")

    visible = (sample.assign(file=[f"{u}{Path(p).suffix.lower()}"
                                   for u, p in zip(sample.uid, sample.local_path)])
               .sort_values("position"))
    payload = {
        "sample_seed": SEED,
        "order_seed": ORDER_SEED,
        "n": len(visible),
        "maps": [{"uid": u, "file": f} for u, f in zip(visible.uid, visible.file)],
    }
    body = json.dumps(payload, ensure_ascii=False, indent=1)
    (args.out / "maps.json").write_text(body, encoding="utf-8")
    (args.out / "maps.js").write_text(f"window.AUDIT_MAPS = {body};\n",
                                      encoding="utf-8")

    print(f"\nwrote sample.csv ({len(sample)} maps), maps.json, maps.js "
          f"and {len(sample)} images in {images}")
    print("NOTE: sample.csv holds the strata and MUST NOT reach the coding tool.")


if __name__ == "__main__":
    main()
