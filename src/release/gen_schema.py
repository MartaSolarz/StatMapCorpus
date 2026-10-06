#!/usr/bin/env python3

import argparse, csv, datetime, hashlib, json, os, sys

import pyarrow.parquet as pq

NOTES = {
    "download_images.py":     "script: retrieves the source images and verifies each download against sha256 / phash (MIT)",
    "positive_keywords.json": "Stage 1 dictionary: map-indicative phrases and their weights, matched against the MapPool alt-text",
    "negative_keywords.json": "Stage 1 dictionary: counter-indicative phrases and their weights",
    "README.md":              "what the corpus is, what changed from version 1, how it was built, how to fetch the images",
    "CODEBOOK.md":            "every column: type, observed values, caveats for analysis",
    "NOTICE.md":              "attribution, derivation chain, licensing, removal procedure",
    "DATASHEET.md":           "motivation, composition, collection process, limitations",
    "VALIDATION.md":          "results of the technical validation",
    "statmapcorpus_dataset.parquet": "the main table: one row per map; identifiers and provenance, image properties and checksums, labels",
    "candidates.parquet":  "all Stage 3 candidates with the outcome of each step (see CODEBOOK.md)",
    "validation.csv":      "human and repeat-run labels of the technical validation (see CODEBOOK.md)",
}

def table_shape(path):
    if path.endswith(".parquet"):
        f = pq.ParquetFile(path)
        return f.metadata.num_rows, [{"name": x.name, "dtype": str(x.type)}
                                     for x in f.schema_arrow]
    if path.endswith(".csv"):
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = sum(1 for _ in reader)
        return rows, [{"name": x, "dtype": "string"} for x in header]
    return None, None


def sort_key(name):
    ext = os.path.splitext(name)[1]
    return ({".parquet": 0, ".npz": 1, ".py": 2, ".json": 3, ".md": 4}.get(ext, 5), name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, help="package root")
    a = ap.parse_args()

    root = os.path.abspath(a.package)
    schema_path = os.path.join(root, "schema.json")
    if not os.path.exists(schema_path):
        sys.exit(f"{schema_path} not found")

    old = json.load(open(schema_path))
    kept = {e["file"]: e for e in old.get("files", [])}

    names = sorted((f for f in os.listdir(root)
                    if not f.startswith(".") and f != "schema.json"
                    and os.path.isfile(os.path.join(root, f))), key=sort_key)

    files = []
    for name in names:
        raw = open(os.path.join(root, name), "rb").read()
        prev = kept.get(name, {})
        entry = {"file": name, "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest(),
                 "note": prev.get("note") or NOTES.get(name, "")}
        for k in ("rows", "columns"):
            if k in prev:
                entry[k] = prev[k]
        if "rows" not in entry or "columns" not in entry:
            rows, columns = table_shape(os.path.join(root, name))
            if rows is not None:
                entry.setdefault("rows", rows)
                entry.setdefault("columns", columns)
        order = ["file", "bytes", "rows", "sha256", "note", "columns"]
        files.append({k: entry[k] for k in order if k in entry})

        changed = prev.get("sha256") not in (None, entry["sha256"])
        print(f"{'CHANGED' if changed else '  same '}  {name:36s} "
              f"{entry['bytes']:>10,} B  {entry['sha256'][:16]}...")

    out = {
        "package": old.get("package", os.path.basename(root)),
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "note": ("schema.json is not listed below: a checksum file cannot carry its own "
                 "checksum. Verify the other files with: shasum -a 256 <file>"),
        "files": files,
    }
    with open(schema_path, "w") as f:
        f.write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")

    missing = [e["file"] for e in files if not e["note"]]
    if missing:
        print(f"\nno note for: {', '.join(missing)} -- add one in schema.json or in NOTES")
    print(f"\n{len(files)} files written to {schema_path}")


if __name__ == "__main__":
    main()
