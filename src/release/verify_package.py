#!/usr/bin/env python3

import argparse, hashlib, json, os, sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OK, BAD = "  OK  ", " FAIL "


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=".", help="package root (contains data/)")
    a = ap.parse_args()
    root = os.path.abspath(a.package)
    data = os.path.join(root, "data")
    if not os.path.isdir(data):
        data = root
    schema_path = os.path.join(data, "schema.json")

    if not os.path.exists(schema_path):
        sys.exit(f"{schema_path} not found -- point --package at the package root")
    schema = json.load(open(schema_path))
    print(f"package: {schema['package']}   exported: {schema['generated_at']}\n")

    failed = 0

    for entry in schema["files"]:
        p = os.path.join(data, entry["file"])
        if not os.path.exists(p):
            print(f"[{BAD}] {entry['file']}: FILE MISSING")
            failed += 1
            continue
        got = sha256_of(p)
        if got != entry["sha256"]:
            print(f"[{BAD}] {entry['file']}: sha256 mismatch")
            print(f"         expected {entry['sha256']}")
            print(f"         got      {got}")
            failed += 1
        else:
            extent = f"{entry['rows']:,} rows" if "rows" in entry else f"{len(open(p,'rb').read()):,} bytes"
            print(f"[{OK}] {entry['file']}: sha256 matches, {extent}")

    if failed:
        print(f"\n{failed} file(s) failed verification -- stopping before consistency checks")
        return 1

    print()
    parquets = [f for f in os.listdir(data) if f.endswith(".parquet")]

    def check(label, cond):
        nonlocal failed
        print(f"[{OK if cond else BAD}] {label}")
        if not cond:
            failed += 1

    if any("statmapcorpus_dataset" in f for f in parquets):
        df = pd.read_parquet(os.path.join(
            data, next(f for f in parquets if "statmapcorpus_dataset" in f)))
        check("uid is unique", df.uid.is_unique)
        check("sha256 present for every record", df.sha256.notna().all())
        check("phash is 16 hex chars everywhere",
              df.phash.notna().all() and (df.phash.str.len() == 16).all())
        check("methods_count >= 1 (dataset definition)", (df.methods_count >= 1).all())
        check("methods_count equals the number of method_* flags set",
              (df.methods_count == df[[c for c in df.columns
                                       if c.startswith("method_")]].sum(axis=1)).all())
        check("legend_is_classed is boolean where present",
              df.legend_is_classed.dropna().isin([True, False]).all())
        check("legend_is_classed is only set for maps that can have a legend class",
              bool((df.legend_is_classed.notna() & (df.methods_count < 1)).sum() == 0))
        md = pq.read_schema(os.path.join(
            data, next(f for f in parquets if "statmapcorpus_dataset" in f))).metadata or {}
        check("inclusion criteria recorded in file metadata",
              b"inclusion_criteria" in md and b"source_corpus" in md)
        check("is_duplicate_byte set only for records that have a group",
              bool((df.is_duplicate_byte.fillna(False) &
                    df.duplicate_group_id.isna()).sum() == 0))
        check("grouped records split exactly into keepers + flagged copies",
              int(df.duplicate_group_id.notna().sum()) ==
              int(df.duplicate_group_id.nunique()) +
              int(df.is_duplicate_byte.fillna(False).sum()))
        check("every duplicate group has exactly one keeper",
              df[df.duplicate_group_id.notna()]
                .groupby("duplicate_group_id")
                .is_duplicate_byte.apply(lambda s: (s == False).sum() == 1).all())

        npz = os.path.join(data, "embeddings_clip_vitl14.npz")
        if os.path.exists(npz):
            z = np.load(npz)
            check("embeddings: uids match the dataset",
                  set(z["uids"].tolist()) == set(df.uid))
            check("embeddings: shape (n, 768) float32",
                  z["emb"].shape == (len(df), 768) and z["emb"].dtype == np.float32)

        cand = os.path.join(data, "candidates.parquet")
        if os.path.exists(cand):
            c = pd.read_parquet(cand, columns=["uid", "in_corpus"])
            check("candidates: uid is unique", c.uid.is_unique)
            check("candidates: in_corpus marks exactly the records of the main table",
                  set(c.loc[c.in_corpus, "uid"]) == set(df.uid))
            val = os.path.join(data, "validation.csv")
            if os.path.exists(val):
                v = pd.read_csv(val, dtype=str)
                check("validation: every uid is a candidate", set(v.uid) <= set(c.uid))

    print(f"\n{'ALL CHECKS PASSED' if not failed else str(failed) + ' CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
