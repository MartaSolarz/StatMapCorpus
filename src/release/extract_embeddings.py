import sqlite3, os, sys, time
import pyarrow.parquet as pq

DB = os.environ.get("STATMAP_DB", "pipeline.db")
SRC = os.environ.get("MAPPOOL_DATA", "mappool_parts")
BATCH_ROWS = 32768

DATASET_WHERE = """
    mt_size_status='pass' AND mt_step3_status='done' AND mt_methods_count>=1
    AND mt_map_language='en' AND mt_is_map=1 AND mt_is_map_dominant=1
    AND mt_is_statistical_map=1 AND mt_has_admin_units=1 AND mt_has_quantitative_data=1
"""

con = sqlite3.connect(DB)
con.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        uid        TEXT PRIMARY KEY,
        embeddings BLOB NOT NULL      -- 768 x float32, little-endian (source precision)
    )
""")
con.commit()

want = {r[0] for r in con.execute(f"SELECT uid FROM candidates WHERE {DATASET_WHERE}")}
have = {r[0] for r in con.execute("SELECT uid FROM embeddings")}
todo = want - have
print(f"DATASET: {len(want)}   already stored: {len(have)}   to find: {len(todo)}")
if not todo:
    print("nothing to do")
    sys.exit(0)

if not os.path.isdir(SRC):
    sys.exit(f"ABORTED: {SRC} not found -- connect the drive")

parts = sorted(f for f in os.listdir(SRC) if f.endswith(".parquet"))
print(f"source files: {len(parts)}")

found, t0 = 0, time.time()
for pi, part in enumerate(parts, 1):
    if not todo:
        break
    path = os.path.join(SRC, part)
    try:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=["uid", "l14_img"]):
            uids = batch.column("uid").to_pylist()
            idx = [i for i, u in enumerate(uids) if u in todo]
            if not idx:
                continue
            arr = (batch.column("l14_img").flatten()
                   .to_numpy(zero_copy_only=False)
                   .reshape(len(uids), 768).astype("<f4", copy=False))
            rows = []
            for i in idx:
                rows.append((uids[i], arr[i].tobytes()))
                todo.discard(uids[i])
            con.executemany("INSERT OR REPLACE INTO embeddings VALUES (?,?)", rows)
            con.commit()
            found += len(rows)
            if not todo:
                break
    except OSError as e:
        sys.exit(f"\nABORTED: drive read error ({e}). Found {found}; "
                 f"reconnect the drive and re-run -- this resumes.")

    el = time.time() - t0
    print(f"  [{pi}/{len(parts)}] {part}: {found} found so far, "
          f"{len(todo)} missing, {el/60:.1f} min", flush=True)

n = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
size = con.execute("SELECT SUM(LENGTH(embeddings)) FROM embeddings").fetchone()[0] or 0
print(f"""
--- REPORT ---
rows in embeddings:   {n}
DATASET coverage:     {n}/{len(want)}  ({n/len(want)*100:.1f}%)
not found:            {len(todo)}
BLOB size:            {size/1e6:.1f} MB
elapsed:              {(time.time()-t0)/60:.1f} min
""")
if todo:
    print("examples of uids not found:", list(todo)[:5])
