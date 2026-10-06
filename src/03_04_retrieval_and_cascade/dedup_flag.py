import sqlite3, hashlib, os
from collections import Counter

DB = os.environ.get("STATMAP_DB", "pipeline.db")

con = sqlite3.connect(DB)

cols = [r[1] for r in con.execute("PRAGMA table_info(candidates)")]
if "is_duplicated" not in cols:
    con.execute("ALTER TABLE candidates ADD COLUMN is_duplicated INTEGER DEFAULT 0")
    con.commit()
    print("column is_duplicated added")
else:
    print("column is_duplicated already exists; resetting it to 0")
    con.execute("UPDATE candidates SET is_duplicated=0")
    con.commit()


def depth(r):
    if r["s6b"] == "done": return 6
    if r["s6a"] == "done": return 5
    if r["s5"]  == "done": return 4
    if r["s4p"] is not None: return 3
    if r["s3"]  == "done": return 3
    if r["s2"]  == "done": return 2
    if r["s1"]  == "done": return 1
    return 0


groups = con.execute("""
    SELECT image_width,image_height,image_size_bytes
    FROM candidates WHERE download_status='success' AND image_size_bytes IS NOT NULL
    GROUP BY image_width,image_height,image_size_bytes HAVING COUNT(*)>1
""").fetchall()

flagged = 0
true_groups = 0
resolved_majority = 0
resolved_furthest = 0
total_files_hashed = 0

for w, h, sz in groups:
    rows = con.execute("""
        SELECT uid, local_path, image_width, image_height,
               mt_step1_status s1, mt_step2_status s2, mt_step3_status s3,
               mt_step4_passed s4p, mt_step5_status s5, mt_step5_all_passed s5pass,
               mt_step6a_status s6a, mt_step6b_status s6b
        FROM candidates
        WHERE download_status='success' AND image_width=? AND image_height=? AND image_size_bytes=?
    """, (w, h, sz)).fetchall()
    colnames = ["uid","local_path","w","h","s1","s2","s3","s4p","s5","s5pass","s6a","s6b"]
    dicts = [dict(zip(colnames, r)) for r in rows]

    by_md5 = {}
    for d in dicts:
        p = d["local_path"]
        if not p or not os.path.exists(p):
            continue
        try:
            m = hashlib.md5(open(p, "rb").read()).hexdigest()
            total_files_hashed += 1
        except Exception:
            continue
        by_md5.setdefault(m, []).append(d)

    for md5, members in by_md5.items():
        if len(members) < 2:
            continue
        true_groups += 1
        maxd = max(depth(m) for m in members)
        top = [m for m in members if depth(m) == maxd]
        if len(top) > 1:
            votes = Counter(m["s5pass"] for m in top if m["s5pass"] is not None)
            if votes:
                maj = votes.most_common(1)[0][0]
                maj_side = [m for m in top if m["s5pass"] == maj]
                if maj_side:
                    top = maj_side
                    resolved_majority += 1
                else:
                    resolved_furthest += 1
            else:
                resolved_furthest += 1
        else:
            resolved_furthest += 1
        keeper = sorted(top, key=lambda m: (-(m["w"] * m["h"]), m["uid"]))[0]
        for m in members:
            if m["uid"] != keeper["uid"]:
                con.execute("UPDATE candidates SET is_duplicated=1 WHERE uid=?", (m["uid"],))
                flagged += 1

con.commit()
con.close()

print(f"files hashed (MD5):        {total_files_hashed:,}")
print(f"true duplicate groups:     {true_groups:,}")
print(f"flagged (is_duplicated=1): {flagged:,}")
print(f"keeper by majority:        {resolved_majority}")
print(f"keeper by depth/tiebreak:  {resolved_furthest}")
