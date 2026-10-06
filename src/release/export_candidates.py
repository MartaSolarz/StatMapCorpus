import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DB = os.environ.get("STATMAP_DB", "pipeline.db")
OUT = os.environ.get("STATMAP_OUT", str(ROOT / "build" / "packages"))
VER = "v2"
FILE = "candidates.parquet"

DATASET_WHERE = """
    mt_size_status='pass' AND mt_step3_status='done' AND mt_methods_count>=1
    AND mt_map_language='en' AND mt_is_map=1 AND mt_is_map_dominant=1
    AND mt_is_statistical_map=1 AND mt_has_admin_units=1 AND mt_has_quantitative_data=1
"""

QUERY = f"""
SELECT
    uid, url, domain, score AS keyword_score, pred_proba,
    url_status, url_http_code, download_status,
    image_width, image_height, image_format,
    mt_size_status, mt_step1_status, mt_is_map, mt_is_map_dominant,
    mt_step2_status, mt_is_statistical_map, mt_has_admin_units,
    mt_has_quantitative_data, mt_map_language, mt_has_text,
    mt_step3_status, mt_methods_count,
    CASE WHEN {DATASET_WHERE} THEN 1 ELSE 0 END AS in_corpus
FROM candidates
ORDER BY uid
"""

EXPECTED = {
    "rows": 203_371,
    "url_status": {"alive": 122_963, "dead": 80_277, "error": 131},
    "download_status": {"success": 116_573, "not_image": 5_770, "failed": 620},
    "gate1_pass": {True: 68_035, False: 48_538},
    "gate2_pass": {True: 67_273, False: 762},
    "gate2_not_a_map": 337,
    "gate2_not_dominant": 425,
    "gate3_pass": {True: 23_646, False: 43_627},
    "gate3_language_only": 17_643,
    "gate4_pass": {True: 23_549, False: 97},
    "in_corpus": 23_549,
}


def nullable_bool(series: pd.Series) -> pd.Series:
    return series.map({1: True, 0: False}).astype("boolean")


def build(con: sqlite3.Connection) -> pd.DataFrame:
    raw = pd.read_sql_query(QUERY, con)
    downloaded = raw.download_status.eq("success")
    out = pd.DataFrame({
        "uid": raw.uid,
        "url": raw.url,
        "domain": raw.domain,
        "keyword_score": raw.keyword_score,
        "pred_proba": raw.pred_proba,
        "url_status": raw.url_status,
        "url_http_code": raw.url_http_code.astype("Int64"),
        "download_status": raw.download_status.where(raw.download_status != "pending"),
        "image_width": raw.image_width.where(downloaded).astype("Int64"),
        "image_height": raw.image_height.where(downloaded).astype("Int64"),
        "image_format": raw.image_format.where(downloaded),
    })

    out["gate1_pass"] = raw.mt_size_status.map(
        {"pass": True, "fail_too_small": False}).astype("boolean")

    reached2 = raw.mt_step1_status.eq("done")
    out["gate2_is_map"] = nullable_bool(raw.mt_is_map.where(reached2))
    out["gate2_map_dominant"] = nullable_bool(raw.mt_is_map_dominant.where(reached2))
    out["gate2_pass"] = (out.gate2_is_map & out.gate2_map_dominant).where(reached2)

    reached3 = raw.mt_step2_status.eq("done")
    out["gate3_is_statistical"] = nullable_bool(raw.mt_is_statistical_map.where(reached3))
    out["gate3_admin_units"] = nullable_bool(raw.mt_has_admin_units.where(reached3))
    out["gate3_quantitative"] = nullable_bool(raw.mt_has_quantitative_data.where(reached3))
    out["gate3_has_text"] = nullable_bool(raw.mt_has_text.where(reached3))
    out["gate3_language"] = raw.mt_map_language.where(reached3)
    out["gate3_pass"] = (out.gate3_is_statistical & out.gate3_admin_units
                         & out.gate3_quantitative
                         & out.gate3_language.eq("en").astype("boolean")).where(reached3)

    reached4 = raw.mt_step3_status.eq("done")
    out["gate4_n_types"] = raw.mt_methods_count.where(reached4).astype("Int64")
    out["gate4_pass"] = out.gate4_n_types.ge(1).where(reached4).astype("boolean")

    out["in_corpus"] = raw.in_corpus.astype(bool)
    return out


def check(df: pd.DataFrame) -> list:
    problems = []

    def same(name, got, want):
        if got != want:
            problems.append(f"{name}: got {got}, expected {want}")

    same("rows", len(df), EXPECTED["rows"])
    same("unique uid", df.uid.nunique(), EXPECTED["rows"])
    for col in ("url_status", "download_status"):
        same(col, df[col].value_counts().to_dict(), EXPECTED[col])
    for col in ("gate1_pass", "gate2_pass", "gate3_pass", "gate4_pass"):
        same(col, {bool(k): int(v) for k, v in df[col].value_counts().items()},
             EXPECTED[col])
    same("gate2_not_a_map", int((df.gate2_is_map == False).sum()),
         EXPECTED["gate2_not_a_map"])
    same("gate2_not_dominant",
         int(((df.gate2_is_map == True) & (df.gate2_map_dominant == False)).sum()),
         EXPECTED["gate2_not_dominant"])
    content_ok = (df.gate3_is_statistical & df.gate3_admin_units
                  & df.gate3_quantitative).fillna(False)
    same("gate3_language_only",
         int((content_ok & df.gate3_language.ne("en")).sum()),
         EXPECTED["gate3_language_only"])
    same("in_corpus", int(df.in_corpus.sum()), EXPECTED["in_corpus"])
    same("in_corpus == gate4_pass",
         bool((df.in_corpus == df.gate4_pass.fillna(False)).all()), True)
    same("gate2 only after gate1",
         int((df.gate2_pass.notna() & (df.gate1_pass != True)).sum()), 0)
    same("gate3 only after gate2",
         int((df.gate3_pass.notna() & (df.gate2_pass != True)).sum()), 0)
    same("gate4 only after gate3",
         int((df.gate4_pass.notna() & (df.gate3_pass != True)).sum()), 0)
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export candidates.parquet: every Stage 3 candidate with the outcome of each step")
    parser.add_argument("--out", type=Path, default=Path(OUT) / f"statmapcorpus-{VER}")
    args = parser.parse_args()

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = build(con)
    con.close()

    problems = check(df)
    if problems:
        raise SystemExit("counts differ from Table 2 -- nothing written:\n  "
                         + "\n  ".join(problems))

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / FILE
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        b"dataset": f"StatMapCorpus {VER} - Stage 3 candidates".encode(),
        b"gates": (b"Gate 1 size (shorter side >= 400 px); Gate 2 map and dominant "
                   b"(Claude Haiku 4.5); Gate 3 statistical map, administrative or "
                   b"statistical units, quantitative data, English (Claude Haiku 4.5); "
                   b"Gate 4 at least one of seven thematic map types (Claude Sonnet 4.6). "
                   b"Null = gate not reached."),
        b"images_included": b"false",
    })
    pq.write_table(table, path, compression="zstd")

    print(f"wrote {path}  ({len(df):,} rows x {len(df.columns)} columns, "
          f"{path.stat().st_size / 1e6:.1f} MB)")
    print("all counts match Table 2")
    for col in ("url_status", "download_status", "gate1_pass", "gate2_pass",
                "gate3_pass", "gate4_pass", "in_corpus"):
        counts = df[col].value_counts(dropna=False).to_dict()
        print(f"  {col:16} {counts}")
    content_ok = (df.gate3_is_statistical & df.gate3_admin_units
                  & df.gate3_quantitative).fillna(False)
    lang = df.loc[content_ok & df.gate3_language.ne("en"), "gate3_language"]
    print(f"  gate3 rejected for language alone: {len(lang):,} "
          f"(unknown {int((lang == 'unknown').sum()):,}, "
          f"another language {int((lang != 'unknown').sum()):,})")


if __name__ == "__main__":
    main()
