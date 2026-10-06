import sqlite3, os, sys, time
import pyarrow.parquet as pq

DB = os.environ.get("STATMAP_DB", "pipeline.db")
SRC = os.environ.get("MAPPOOL_DATA", "mappool_parts")
BATCH_ROWS = 262144

DATASET_WHERE = """
    mt_size_status='pass' AND mt_step3_status='done' AND mt_methods_count>=1
    AND mt_map_language='en' AND mt_is_map=1 AND mt_is_map_dominant=1
    AND mt_is_statistical_map=1 AND mt_has_admin_units=1 AND mt_has_quantitative_data=1
"""

con = sqlite3.connect(DB)
cols = [r[1] for r in con.execute("PRAGMA table_info(candidates)")]
for name in ("pos_keys", "neg_keys"):
    if name not in cols:
        con.execute(f"ALTER TABLE candidates ADD COLUMN {name} TEXT")
        print(f"added column {name}")
con.commit()

want = {r[0] for r in con.execute(f"SELECT uid FROM candidates WHERE {DATASET_WHERE}")}
todo = {r[0] for r in con.execute(
    f"SELECT uid FROM candidates WHERE {DATASET_WHERE} AND pos_keys IS NULL")}
print(f"DATASET: {len(want)}   to find: {len(todo)}")
if not todo:
    print("nothing to do")
    sys.exit(0)

if not os.path.isdir(SRC):
    sys.exit(f"ABORTED: {SRC} not found -- connect the drive")

parts = sorted(f for f in os.listdir(SRC) if f.endswith(".parquet"))
found, t0 = 0, time.time()

for pi, part in enumerate(parts, 1):
    if not todo:
        break
    try:
        pf = pq.ParquetFile(os.path.join(SRC, part))
        for batch in pf.iter_batches(batch_size=BATCH_ROWS,
                                     columns=["uid", "pos_keys", "neg_keys"]):
            uids = batch.column("uid").to_pylist()
            idx = [i for i, u in enumerate(uids) if u in todo]
            if not idx:
                continue
            pos = batch.column("pos_keys").to_pylist()
            neg = batch.column("neg_keys").to_pylist()
            rows = []
            for i in idx:
                rows.append((pos[i] or "", neg[i] or "", uids[i]))
                todo.discard(uids[i])
            con.executemany(
                "UPDATE candidates SET pos_keys=?, neg_keys=? WHERE uid=?", rows)
            con.commit()
            found += len(rows)
            if not todo:
                break
    except OSError as e:
        sys.exit(f"\nABORTED: drive read error ({e}). Found {found}; re-run to resume.")

    print(f"  [{pi}/{len(parts)}] {part}: {found} found, {len(todo)} missing, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

n = con.execute(
    f"SELECT COUNT(*) FROM candidates WHERE {DATASET_WHERE} AND pos_keys IS NOT NULL"
).fetchone()[0]
nonempty = con.execute(
    f"SELECT COUNT(*) FROM candidates WHERE {DATASET_WHERE} AND pos_keys <> ''"
).fetchone()[0]
print(f"""
--- REPORT ---
keywords stored:      {n}/{len(want)}  ({n/len(want)*100:.1f}%)
with a positive match:{nonempty:>7}  ({nonempty/max(n,1)*100:.1f}%)
not found:            {len(todo)}
elapsed:              {(time.time()-t0)/60:.1f} min
""")
