import argparse
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

PIPELINE_DIR = Path(os.environ.get(
    "STATMAP_PIPELINE_DIR", "src/03_04_retrieval_and_cascade")).resolve()
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config as pipeline_config
from multiturn import config as mt_config
from multiturn import pipeline as mt

PROD_DB = Path(os.environ.get("STATMAP_DB", "pipeline.db"))
RETEST_DB = HERE / "runs.sqlite"
SAMPLE = HERE / "sample_uids.csv"
RESERVE = HERE / "sample_uids_reserve.csv"

MODEL_GATE3 = "claude-haiku-4-5-20251001"
MODEL_GATE4 = "claude-sonnet-4-6"

PRICES = {
    MODEL_GATE3: (1.0e-6, 5.0e-6, 0.10e-6, 1.25e-6),
    MODEL_GATE4: (3.0e-6, 15.0e-6, 0.30e-6, 3.75e-6),
}
COST_LIMIT_USD = 25.00
COST_STOP_USD = 23.00

METHOD_FIELDS = mt._METHOD_FIELDS
GATE3_FIELDS = ("map_language", "is_statistical_map", "has_admin_units",
                "has_quantitative_data", "has_text", "data_level")


def cost_usd(model, in_tok, out_tok, cache_read, cache_write):
    pi, po, pr, pw = PRICES[model]
    return in_tok * pi + out_tok * po + cache_read * pr + cache_write * pw


DDL = """
CREATE TABLE IF NOT EXISTS runs (
    uid           TEXT NOT NULL,
    run           TEXT NOT NULL,
    gate          TEXT NOT NULL,
    model         TEXT,
    ts_utc        TEXT,
    status        TEXT,
    error         TEXT,
    raw_response  TEXT,
    parsed_json   TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cache_read    INTEGER,
    cache_creation INTEGER,
    cost_usd      REAL,
    elapsed_s     REAL,
    layer         TEXT,
    substituted_for TEXT,
    PRIMARY KEY (uid, run, gate)
);
"""


def open_store():
    conn = sqlite3.connect(RETEST_DB, check_same_thread=False)
    conn.executescript(DDL)
    conn.commit()
    return conn


def already_done(conn):
    cur = conn.execute("SELECT uid, run, gate FROM runs WHERE status = 'done'")
    return {tuple(r) for r in cur.fetchall()}


def save(conn, lock, row):
    with lock:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (:uid,:run,:gate,:model,:ts_utc,:status,:error,"
            ":raw_response,:parsed_json,:input_tokens,:output_tokens,:cache_read,:cache_creation,"
            ":cost_usd,:elapsed_s,:layer,:substituted_for)", row)
        conn.commit()


def load_inputs(uids):
    con = sqlite3.connect(f"file:{PROD_DB}?mode=ro", uri=True)
    placeholders = ",".join("?" * len(uids))
    frame = pd.read_sql(
        f"""SELECT uid, local_path, mt_step1_raw_response, mt_step2_raw_response,
                   mt_step2_status, mt_step3_status
            FROM candidates WHERE uid IN ({placeholders})""", con, params=list(uids))
    con.close()
    return frame.set_index("uid")


class Budget:

    def __init__(self, spent=0.0):
        self._spent = spent
        self._lock = threading.Lock()
        self.stopped = False

    def add(self, amount):
        with self._lock:
            self._spent += amount
            if self._spent >= COST_STOP_USD:
                self.stopped = True
            return self._spent

    @property
    def spent(self):
        with self._lock:
            return self._spent


def do_one(client, uid, meta, run, gates, conn, lock, budget):
    out = []
    for gate in gates:
        if budget.stopped:
            break
        model = MODEL_GATE3 if gate == "gate3" else MODEL_GATE4
        t0 = time.monotonic()
        if gate == "gate3":
            res = mt._process_step2_one(client, model, uid, meta["local_path"],
                                        meta["mt_step1_raw_response"])
            parsed = {k: res.get(k) for k in GATE3_FIELDS}
        else:
            res = mt._process_step3_one(client, model, uid, meta["local_path"],
                                        meta["mt_step1_raw_response"],
                                        meta["mt_step2_raw_response"])
            parsed = {k: res.get(k) for k in METHOD_FIELDS}
            parsed["legend_is_classed"] = res.get("legend_is_classed")
            parsed["confidence"] = res.get("confidence")
            parsed["methods_count"] = res.get("methods_count")
        elapsed = time.monotonic() - t0
        c = cost_usd(model, res["input_tokens"], res["output_tokens"],
                     res["cache_read"], res["cache_creation"])
        budget.add(c)
        save(conn, lock, {
            "uid": uid, "run": run, "gate": gate, "model": model,
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": res["status"], "error": res["error"],
            "raw_response": res["raw_response"],
            "parsed_json": json.dumps(parsed, ensure_ascii=False) if res["status"] == "done" else None,
            "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
            "cache_read": res["cache_read"], "cache_creation": res["cache_creation"],
            "cost_usd": c, "elapsed_s": round(elapsed, 2),
            "layer": meta["layer"], "substituted_for": meta.get("substituted_for"),
        })
        out.append((gate, res["status"]))
    return uid, out


def resolve_sample(allow_substitute):
    sample = pd.read_csv(SAMPLE)
    inputs = load_inputs(sample["uid"].tolist())

    missing_db = [u for u in sample["uid"] if u not in inputs.index]
    if missing_db:
        raise SystemExit(f"STOP: {len(missing_db)} uids are not in {PROD_DB}: "
                         f"{missing_db[:5]}")

    rows, substitutions = [], []
    reserve = pd.read_csv(RESERVE)["uid"].tolist() if RESERVE.exists() else []
    reserve_inputs = load_inputs(reserve) if reserve else pd.DataFrame()
    reserve_iter = iter(reserve)

    for uid, layer in zip(sample["uid"], sample["layer"]):
        meta = inputs.loc[uid].to_dict()
        if Path(meta["local_path"]).exists():
            rows.append({"uid": uid, "layer": layer, "substituted_for": None, **meta})
            continue
        if not allow_substitute:
            raise SystemExit(
                f"STOP: no image file for {uid} ({meta['local_path']}). "
                "Mount the image cache. Deliberate substitution: --allow-substitute.")
        for candidate in reserve_iter:
            cmeta = reserve_inputs.loc[candidate].to_dict()
            if Path(cmeta["local_path"]).exists():
                rows.append({"uid": candidate, "layer": layer,
                             "substituted_for": uid, **cmeta})
                substitutions.append((uid, candidate))
                break
        else:
            raise SystemExit(f"STOP: reserve exhausted at {uid}")

    if substitutions:
        print(f"WARNING: {len(substitutions)} maps substituted from the reserve: "
              f"{substitutions}")
    return rows


def check_models(client):
    available = {m.id for m in client.models.list(limit=100).data}
    for model in (MODEL_GATE3, MODEL_GATE4):
        if model not in available:
            raise SystemExit(f"STOP: model {model} unavailable. "
                             "Substituting a model is forbidden.")
    print(f"models OK: {MODEL_GATE3}, {MODEL_GATE4}")


def export_parquet(conn):
    frame = pd.read_sql("SELECT * FROM runs", conn)
    frame.to_parquet(HERE / "runs.parquet", index=False)
    print(f"export: runs.parquet ({len(frame)} rows)")


def main():
    ap = argparse.ArgumentParser(description="Re-run Gates 3 and 4 on the test-retest sample")
    ap.add_argument("--runs", nargs="+", default=["run1", "run2"],
                    choices=["run1", "run2"])
    ap.add_argument("--gates", nargs="+", default=["gate3", "gate4"],
                    choices=["gate3", "gate4"])
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N maps (smoke test)")
    ap.add_argument("--workers", type=int, default=mt_config.WORKERS)
    ap.add_argument("--allow-substitute", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    conn = open_store()
    if args.export_only:
        export_parquet(conn)
        return

    if not Path(pipeline_config.IMAGE_CACHE_DIR).exists():
        raise SystemExit(f"STOP: image directory {pipeline_config.IMAGE_CACHE_DIR} "
                         "unavailable.")

    rows = resolve_sample(args.allow_substitute)
    if args.limit:
        rows = rows[:args.limit]
    print(f"sample: {len(rows)} maps, runs: {args.runs}, gates: {args.gates}")

    client = mt._get_client()
    check_models(client)

    spent_before = conn.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM runs").fetchone()[0]
    budget = Budget(spent_before)
    if spent_before:
        print(f"cost already incurred in earlier invocations: ${spent_before:.2f}")

    done = already_done(conn)
    lock = threading.Lock()
    t_start = time.monotonic()
    failures = []

    for run in args.runs:
        todo = [r for r in rows
                if any((r["uid"], run, g) not in done for g in args.gates)]
        print(f"\n=== {run}: {len(todo)} maps to do "
              f"({len(rows) - len(todo)} already finished) ===")
        if not todo:
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(do_one, client, r["uid"], r, run,
                            [g for g in args.gates if (r["uid"], run, g) not in done],
                            conn, lock, budget): r["uid"]
                for r in todo
            }
            for i, fut in enumerate(as_completed(futures), 1):
                uid, results = fut.result()
                for gate, status in results:
                    if status != "done":
                        failures.append((uid, run, gate, status))
                if i % 25 == 0 or i == len(futures):
                    print(f"  [{run}] {i}/{len(futures)}  cost ${budget.spent:.2f}  "
                          f"errors {len(failures)}")
                if budget.stopped:
                    print(f"\nSTOP: cost threshold ${COST_STOP_USD:.2f} reached "
                          f"(limit {COST_LIMIT_USD:.2f}). State saved, resumable.")
                    break
        if budget.stopped:
            break

    export_parquet(conn)
    print(f"\ntotal cost: ${budget.spent:.4f} / limit ${COST_LIMIT_USD:.2f}")
    print(f"time: {(time.monotonic() - t_start) / 60:.1f} min")
    print(f"failures: {len(failures)}")
    for failure in failures[:20]:
        print("   ", failure)


if __name__ == "__main__":
    main()
