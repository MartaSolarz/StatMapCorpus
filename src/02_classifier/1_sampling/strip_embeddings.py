import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

STAGE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = Path(os.environ.get("MAPPOOL_DATA", "mappool_parts"))
OUTPUT_FILE = STAGE_DIR / "data" / "sample" / "all_no_embeddings.parquet"


def main():
    files = sorted(INPUT_DIR.glob("part*.parquet"))
    if not files:
        raise SystemExit(f"no part*.parquet files in {INPUT_DIR}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    rows = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
            table = pa.Table.from_batches([batch])
            if "l14_img" in table.schema.names:
                table = table.drop(["l14_img"])
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_FILE, table.schema, compression="gzip")
            writer.write_table(table)
            rows += table.num_rows
        print(f"{path.name}: done ({rows:,} records so far)")
    writer.close()
    print(f"wrote {OUTPUT_FILE} ({rows:,} records)")


if __name__ == "__main__":
    main()
