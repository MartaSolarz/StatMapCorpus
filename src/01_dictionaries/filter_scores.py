import argparse
import glob
import os
from collections import defaultdict

import pyarrow.compute as pc
import pyarrow.parquet as pq

INPUT_FOLDER = "./processed"
OUTPUT_FOLDER = "./processed_min1"
MIN_SCORE = -1.0


def shards_by_part(input_folder, parts):
    files = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(input_folder, "part_*", "*_processed.parquet"))):
        part = os.path.basename(os.path.dirname(path))
        if parts is None or part in parts:
            files[part].append(path)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Keep records with keyword score >= -1.0 and merge them into one file per part")
    parser.add_argument("--input", default=INPUT_FOLDER)
    parser.add_argument("--output", default=OUTPUT_FOLDER)
    parser.add_argument("--parts", default=None, help='e.g. "0,1,2"; default: all parts')
    parser.add_argument("--count-only", action="store_true",
                        help="count records and kept records without writing anything")
    args = parser.parse_args()

    parts = None if not args.parts else {f"part_{p.strip()}" for p in args.parts.split(",")}
    files = shards_by_part(args.input, parts)
    if not args.count_only:
        os.makedirs(args.output, exist_ok=True)

    total_rows = total_kept = 0
    for part, paths in sorted(files.items()):
        rows = kept = 0
        writer = None
        for path in paths:
            if args.count_only:
                score = pq.read_table(path, columns=["score"]).column("score")
                rows += len(score)
                kept += pc.sum(pc.greater_equal(score, MIN_SCORE)).as_py() or 0
                continue
            table = pq.read_table(path)
            selected = table.filter(pc.greater_equal(table["score"], MIN_SCORE))
            rows += table.num_rows
            kept += selected.num_rows
            if selected.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(os.path.join(args.output, f"{part}.parquet"),
                                              selected.schema, compression="zstd")
                writer.write_table(selected)
        if writer is not None:
            writer.close()
        print(f"{part}: {len(paths):,} shards, {rows:,} records, {kept:,} with score >= {MIN_SCORE}")
        total_rows += rows
        total_kept += kept

    print(f"total: {total_rows:,} records, {total_kept:,} kept "
          f"({100 * total_kept / max(total_rows, 1):.1f}%)")


if __name__ == "__main__":
    main()
