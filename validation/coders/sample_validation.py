#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

METHODS = [
    ("M1", "method_choropleth", "choropleth map"),
    ("M2", "method_diagrams", "diagram map"),
    ("M3", "method_isolines", "isarithmic map"),
    ("M4", "method_dot_density", "dot density map"),
    ("M5", "method_heat_map", "heat map"),
    ("M6", "method_cartogram", "cartogram"),
    ("M7", "method_flow_map", "flow map"),
]
CODES = [m[0] for m in METHODS] + ["TXT"]


def fill_stratum(frame, column, need, seed, banned_uids, banned_phash):
    if need <= 0:
        return []
    pool = frame[frame[column] & ~frame.uid.isin(banned_uids)]
    picked = []
    for row in pool.sample(frac=1.0, random_state=seed).itertuples():
        if len(picked) >= need:
            break
        if row.phash in banned_phash:
            continue
        banned_phash.add(row.phash)
        picked.append(row.uid)
    return picked


def build(frame, per_class, seed, state):
    vetoed = set(state["vetoed"])
    seen = set(state["seen"])
    strata = {c: [u for u in state["strata"].get(c, []) if u not in vetoed]
              for c in CODES}
    kept = {u for uids in strata.values() for u in uids}
    banned_phash = set(frame[frame.uid.isin(kept | seen | vetoed)].phash)

    for code, column, _name in METHODS:
        need = per_class - len(strata[code])
        added = fill_stratum(frame, column, need, seed + state["round"],
                             kept | seen | vetoed, banned_phash)
        if added:
            print(f"  {code}: drew {len(added)} more")
        if len(added) < need:
            print(f"  NOTE: {code} - {need - len(added)} short, the pool is exhausted")
        strata[code] += added
        if len(strata[code]) > per_class:
            surplus = strata[code][per_class:]
            strata[code] = strata[code][:per_class]
            print(f"  {code}: dropping {len(surplus)} replacements from an earlier round")

    no_text = [r.uid for r in frame[~frame.has_text].itertuples()
               if r.uid not in vetoed]
    dropped = len(frame[~frame.has_text]) - len(no_text)
    strata["TXT"] = no_text
    if dropped:
        print(f"  TXT: {dropped} vetoed, the stratum shrinks to {len(no_text)} "
              f"- this cell cannot be topped up, because that is its whole population")

    seen |= {u for uids in strata.values() for u in uids}
    state["strata"] = strata
    state["seen"] = sorted(seen)
    state["vetoed"] = sorted(vetoed)
    return strata


def main() -> int:
    ap = argparse.ArgumentParser(description="Draw the coder validation sample from the corpus")
    ap.add_argument("--corpus", required=True, help="main corpus table (parquet)")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--veto", help="maps rejected as technically unusable: "
                                   "{\"round\": n, \"veto\": [...]}")
    ap.add_argument("--restore", help="{\"restore\": [...]} - "
                    "lifts a veto and removes the replacement drawn for that map")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    corpus = Path(a.corpus)
    if not corpus.exists():
        print(f"no such file: {corpus}", file=sys.stderr)
        return 1
    full = pd.read_parquet(corpus)
    frame = full[~full.is_duplicate_byte].copy()
    print(f"corpus: {len(full)} maps, without byte duplicates: {len(frame)}")

    out = Path(a.out_dir)
    state_path = out / "sample_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"round": 0, "strata": {}, "seen": [], "vetoed": []}

    if a.veto:
        review = json.loads(Path(a.veto).read_text(encoding="utf-8"))
        fresh = [u for u in review.get("veto", []) if u not in state["vetoed"]]
        known = {u for uids in state["strata"].values() for u in uids}
        unknown = [u for u in fresh if u not in known]
        for u in unknown:
            print(f"  NOTE: veto on {u[:12]}, which is not in the current sample")
        for code, uids in state["strata"].items():
            for u in fresh:
                if u in uids:
                    state.setdefault("vetoed_from", {})[u] = code
        state["vetoed"] = sorted(set(state["vetoed"]) | set(fresh))
        print(f"veto from round {review.get('round', '?')}: {len(fresh)} new "
              f"(rejected in total: {len(state['vetoed'])})")

    if a.restore:
        wanted = json.loads(
            Path(a.restore).read_text(encoding="utf-8")).get("restore", [])
        origin = state.get("vetoed_from", {})
        back = [u for u in wanted if u in state["vetoed"]]
        for u in wanted:
            if u not in state["vetoed"]:
                print(f"  NOTE: {u[:12]} was not rejected - skipping")
            elif u not in origin:
                print(f"  NOTE: do not know which stratum {u[:12]} came from - skipping")
        for u in back:
            code = origin.get(u)
            if code:
                state["strata"].setdefault(code, []).insert(0, u)
        state["vetoed"] = [u for u in state["vetoed"] if u not in back]
        state["restored"] = sorted(set(state.get("restored", [])) | set(back))
        print(f"restored {len(back)} maps "
              f"(still rejected: {len(state['vetoed'])})")

    state["round"] += 1
    print(f"\nROUND {state['round']}")
    strata = build(frame, a.per_class, a.seed, state)

    uids = []
    for code in CODES:
        for uid in strata[code]:
            if uid not in uids:
                uids.append(uid)

    print(f"\ndraws: {sum(len(v) for v in strata.values())}, "
          f"unique maps: {len(uids)}")
    print(f"\n{'cls':4s} {'posit.':>8s} {'sample':>7s} {'hybrid':>7s} {'share':>8s}")
    audit = {}
    for code, column, name in METHODS:
        pool = frame[frame[column]]
        picked = frame[frame.uid.isin(strata[code])]
        hybrid = int((picked.methods_count > 1).sum())
        print(f"{code:4s} {len(pool):8d} {len(picked):7d} {hybrid:7d} "
              f"{100 * hybrid / max(1, len(picked)):7.0f}%")
        audit[code] = {"name": name, "column": column,
                       "positives_in_corpus": int(len(pool)),
                       "drawn": int(len(picked)), "hybrid": hybrid,
                       "uids": strata[code]}
    audit["TXT"] = {"name": "no text", "column": "has_text == false",
                    "positives_in_corpus": int((~frame.has_text).sum()),
                    "drawn": len(strata["TXT"]), "hybrid": None,
                    "uids": strata["TXT"]}

    chosen = frame[frame.uid.isin(uids)]
    legend_counts = Counter(chosen.legend_is_classed.map(
        {True: "classed", False: "continuous"}).fillna("not applicable"))
    print(f"\nlegend type: {dict(legend_counts)}")
    print(f"methods per map: {dict(sorted(Counter(chosen.methods_count).items()))}")
    print(f"model confidence: {dict(Counter(chosen.classification_confidence))}")

    if not a.write:
        print("\n(dry run - add --write to save)")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "coding_uids.txt").write_text("\n".join(uids) + "\n", encoding="utf-8")
    state_path.write_text(json.dumps(state, indent=1, ensure_ascii=False),
                          encoding="utf-8")
    (out / "sample_audit.json").write_text(json.dumps({
        "corpus": str(corpus), "seed": a.seed, "per_class": a.per_class,
        "round": state["round"], "n_corpus": int(len(full)),
        "n_without_duplicates": int(len(frame)), "n_unique": len(uids),
        "vetoed": state["vetoed"], "strata": audit,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out / 'coding_uids.txt'} ({len(uids)} uids)")
    print(f"wrote {out / 'sample_audit.json'} and {state_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
