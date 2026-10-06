import sqlite3, json, os, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DB = os.environ.get("STATMAP_DB", "pipeline.db")
OUT = os.environ.get("STATMAP_OUT", str(ROOT / "build" / "packages"))
VER = "v2"

DATASET_WHERE = """
    mt_size_status='pass' AND mt_step3_status='done' AND mt_methods_count>=1
    AND mt_map_language='en' AND mt_is_map=1 AND mt_is_map_dominant=1
    AND mt_is_statistical_map=1 AND mt_has_admin_units=1 AND mt_has_quantitative_data=1
"""

def as_bool(v):
    return None if v is None else bool(v)


con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
con.row_factory = sqlite3.Row
PKG = f"{OUT}/statmapcorpus-{VER}"
os.makedirs(PKG, exist_ok=True)

stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

print("main table...")
rows = con.execute(f"SELECT * FROM candidates WHERE {DATASET_WHERE} ORDER BY uid").fetchall()
print(f"  records: {len(rows)}")

base = []
for r in rows:
    base.append({
        "uid": r["uid"],
        "url": r["url"],
        "domain": r["domain"],
        "keyword_score": r["score"],
        "pos_keys": r["pos_keys"],
        "neg_keys": r["neg_keys"],
        "pred_proba": r["pred_proba"],
        "image_width": r["image_width"],
        "image_height": r["image_height"],
        "image_max_dim": r["image_max_dim"],
        "image_format": r["image_format"],
        "image_size_bytes": r["image_size_bytes"],
        "sha256": r["sha256"],
        "phash": r["phash"],
        "retrieved_at": r["downloaded_at"],
        "short_description": r["mt_short_description"],
        "has_text": as_bool(r["mt_has_text"]),
        "method_choropleth": as_bool(r["mt_has_choropleth"]),
        "method_diagrams": as_bool(r["mt_has_diagrams"]),
        "method_dot_density": as_bool(r["mt_has_dot_density"]),
        "method_isolines": as_bool(r["mt_has_isolines"]),
        "method_cartogram": as_bool(r["mt_has_cartogram"]),
        "method_flow_map": as_bool(r["mt_has_flow_map"]),
        "method_heat_map": as_bool(r["mt_has_heat_map"]),
        "methods_count": r["mt_methods_count"],
        "legend_is_classed": as_bool(r["mt_legend_is_classed"]),
        "classification_confidence": r["mt_step3_confidence"],
        "is_duplicate_byte": bool(r["duplicate_group_id"]) and bool(r["is_duplicated"]),
        "duplicate_group_id": r["duplicate_group_id"],
        "annotated_at": r["mt_step3_checked_at"],
    })

df1 = pd.DataFrame(base)
p1 = f"{PKG}/statmapcorpus_dataset.parquet"

pipeline_version = con.execute(
    "SELECT mt_pipeline_version FROM candidates WHERE mt_pipeline_version IS NOT NULL LIMIT 1"
).fetchone()[0]
CONSTANT_PROPERTIES = {
    "dataset": f"StatMapCorpus {VER}",
    "source_corpus": "MapPool (Schnurer 2024) / CommonPool / DataComp",
    "pipeline_version": pipeline_version,
    "inclusion_criteria": json.dumps({
        "is_map": True, "is_map_dominant": True, "is_statistical_map": True,
        "has_admin_units": True, "has_quantitative_data": True,
        "map_language": "en", "methods_count": ">=1",
    }),
    "inclusion_note": ("Every record satisfies all of the above; they were verified per "
                       "record during annotation and are not stored as columns because "
                       "they do not vary. See CODEBOOK.md, 'Properties shared by every "
                       "record'."),
    "images_included": "false",
}
tbl1 = pa.Table.from_pandas(df1, preserve_index=False)
tbl1 = tbl1.cast(pa.schema(
    [f.with_type(pa.large_string()) if pa.types.is_string(f.type) else f
     for f in tbl1.schema], metadata=tbl1.schema.metadata))
tbl1 = tbl1.replace_schema_metadata({
    **(tbl1.schema.metadata or {}),
    **{k.encode(): v.encode() for k, v in CONSTANT_PROPERTIES.items()},
})
pq.write_table(tbl1, p1, compression="zstd")

emb = con.execute("SELECT uid, embeddings FROM embeddings ORDER BY uid").fetchall()
euids = np.array([e["uid"] for e in emb])
emat = np.stack([np.frombuffer(e["embeddings"], dtype="<f4") for e in emb])
p1e = f"{PKG}/embeddings_clip_vitl14.npz"
np.savez_compressed(p1e, uids=euids, emb=emat)
print(f"  embeddings: {emat.shape} {emat.dtype}")

def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def schema_of(df, path, note):
    return {
        "file": os.path.basename(path),
        "rows": len(df),
        "sha256": sha_of(path),
        "note": note,
        "columns": [{"name": c, "dtype": str(t)} for c, t in df.dtypes.items()],
    }


base_sha = sha_of(p1)
s1 = {"package": f"statmapcorpus-{VER}", "generated_at": stamp,
      "files": [schema_of(df1, p1, "annotations, provenance and checksums"),
                {"file": os.path.basename(p1e), "rows": int(emat.shape[0]),
                 "sha256": sha_of(p1e),
                 "note": "CLIP ViT-L/14 image embeddings, float32, source precision "
                         "from MapPool; L2-normalised at source",
                 "columns": [{"name": "uids", "dtype": "str"},
                             {"name": "emb", "dtype": "float32[768]"}]}]}

with open(f"{PKG}/schema.json", "w") as fh:
    json.dump(s1, fh, indent=2, ensure_ascii=False)

mb = lambda p: os.path.getsize(p) / 1e6
print(f"""
--- REPORT ---
statmapcorpus-{VER}
  statmapcorpus_dataset.parquet      {len(df1):>6} x {len(df1.columns):>3} cols  {mb(p1):>6.1f} MB
  embeddings_clip_vitl14.npz           {emat.shape[0]:>6} x 768 f32   {mb(p1e):>6.1f} MB

sha256 of the main table:
  {base_sha}
""")
