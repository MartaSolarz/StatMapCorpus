import sqlite3, hashlib, os, sys, time
import numpy as np
from PIL import Image
from scipy.fftpack import dct

DB = os.environ.get("STATMAP_DB", "pipeline.db")
BATCH = 500

HASH_SIZE = 8
IMG_SIZE = 32

DATASET_WHERE = """
    mt_size_status='pass' AND mt_step3_status='done' AND mt_methods_count>=1
    AND mt_map_language='en' AND mt_is_map=1 AND mt_is_map_dominant=1
    AND mt_is_statistical_map=1 AND mt_has_admin_units=1 AND mt_has_quantitative_data=1
"""


def phash(path):
    im = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    px = np.asarray(im, dtype=np.float64)
    d = dct(dct(px, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:HASH_SIZE, :HASH_SIZE]
    med = np.median(low)
    val = 0
    for b in (low > med).flatten():
        val = (val << 1) | int(b)
    return f"{val:016x}"


def wait_for_drive(path, timeout=120):
    root = os.path.dirname(path)
    while root and root != os.sep and not os.path.isdir(root):
        root = os.path.dirname(root)
    t = time.time()
    while not os.path.isdir(root) or not os.listdir(root):
        if time.time() - t > timeout:
            return False
        print(f"  drive {root} vanished -- waiting...", flush=True)
        time.sleep(5)
    return True


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


con = sqlite3.connect(DB)

info = {r[1]: r[2] for r in con.execute("PRAGMA table_info(candidates)")}
if info.get("phash") == "INTEGER":
    if con.execute("SELECT COUNT(*) FROM candidates WHERE phash IS NOT NULL").fetchone()[0]:
        sys.exit("phash INTEGER column holds data -- aborting, needs manual migration")
    con.execute("ALTER TABLE candidates DROP COLUMN phash")
    info.pop("phash")
    print("dropped empty phash INTEGER column -- will be re-added as TEXT")

for name, typ in [("sha256", "TEXT"), ("phash", "TEXT"), ("duplicate_group_id", "TEXT")]:
    if name not in info:
        con.execute(f"ALTER TABLE candidates ADD COLUMN {name} {typ}")
        print(f"added column {name}")
con.commit()

todo = con.execute(
    f"SELECT uid, local_path FROM candidates WHERE {DATASET_WHERE} AND sha256 IS NULL"
).fetchall()
done_already = con.execute(
    f"SELECT COUNT(*) FROM candidates WHERE {DATASET_WHERE} AND sha256 IS NOT NULL"
).fetchone()[0]
print(f"to compute: {len(todo)}   (already done: {done_already})")

missing = errors = 0
buf, t0 = [], time.time()

for i, (uid, path) in enumerate(todo, 1):
    if not path:
        missing += 1
        continue
    try:
        if not os.path.exists(path):
            if not os.path.isdir(os.path.dirname(path)):
                if not wait_for_drive(path):
                    sys.exit(f"\nABORTED: drive unavailable. Computed {i-1} of {len(todo)}; "
                             f"reconnect the drive and re-run -- this resumes where it stopped.")
                continue
            missing += 1
            continue
        s, p = sha256(path), phash(path)
    except OSError as e:
        if not wait_for_drive(path):
            sys.exit(f"\nABORTED: drive unavailable ({e}). Computed {i-1} of {len(todo)}; "
                     f"reconnect the drive and re-run -- this resumes where it stopped.")
        errors += 1
        print(f"  ERROR {uid}: {type(e).__name__}: {e}")
        continue
    except Exception as e:
        errors += 1
        print(f"  ERROR {uid}: {type(e).__name__}: {e}")
        continue

    buf.append((s, p, uid))
    if len(buf) >= BATCH:
        con.executemany("UPDATE candidates SET sha256=?, phash=? WHERE uid=?", buf)
        con.commit()
        buf.clear()
        el = time.time() - t0
        print(f"  {i}/{len(todo)}  {el/i*1000:.0f} ms/file  "
              f"~{el/i*(len(todo)-i)/60:.1f} min left", flush=True)

if buf:
    con.executemany("UPDATE candidates SET sha256=?, phash=? WHERE uid=?", buf)
    con.commit()

n_missing = con.execute(
    f"SELECT COUNT(*) FROM candidates WHERE {DATASET_WHERE} AND sha256 IS NULL"
).fetchone()[0]
if n_missing:
    con.execute(f"UPDATE candidates SET duplicate_group_id=NULL WHERE {DATASET_WHERE}")
    con.commit()
    sys.exit(f"\nsha256 missing for {n_missing} records -- duplicate_group_id NOT assigned "
             f"(cleared). Finish computing and re-run.")

con.execute(f"UPDATE candidates SET duplicate_group_id=NULL WHERE {DATASET_WHERE}")
con.execute(f"""
    UPDATE candidates SET duplicate_group_id = substr(sha256, 1, 16)
    WHERE {DATASET_WHERE} AND sha256 IN (
        SELECT sha256 FROM candidates WHERE {DATASET_WHERE} AND sha256 IS NOT NULL
        GROUP BY sha256 HAVING COUNT(*) > 1
    )
""")
con.commit()

q = lambda sql: con.execute(sql).fetchone()[0]
n_ds = q(f"SELECT COUNT(*) FROM candidates WHERE {DATASET_WHERE}")
n_sha = q(f"SELECT COUNT(*) FROM candidates WHERE {DATASET_WHERE} AND sha256 IS NOT NULL")
n_grp = q(f"SELECT COUNT(DISTINCT duplicate_group_id) FROM candidates WHERE {DATASET_WHERE} AND duplicate_group_id IS NOT NULL")
n_in_grp = q(f"SELECT COUNT(*) FROM candidates WHERE {DATASET_WHERE} AND duplicate_group_id IS NOT NULL")
n_flag = q(f"SELECT COUNT(*) FROM candidates WHERE {DATASET_WHERE} AND is_duplicated=1")
n_orphan = q(f"SELECT COUNT(*) FROM candidates WHERE {DATASET_WHERE} AND is_duplicated=1 AND duplicate_group_id IS NULL")

print(f"""
--- REPORT ---
DATASET:                  {n_ds}
sha256 + phash computed:  {n_sha}
file missing:             {missing}
read errors:              {errors}

duplicate groups:         {n_grp}
records in groups:        {n_in_grp}   (redundant: {n_in_grp - n_grp})
is_duplicated=1:          {n_flag}
  of which no group:      {n_orphan}   <- twin lies OUTSIDE the DATASET
elapsed:                  {(time.time()-t0)/60:.1f} min
""")
