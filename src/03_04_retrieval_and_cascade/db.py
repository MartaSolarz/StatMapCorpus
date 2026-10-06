import sqlite3
import json
import contextlib
from datetime import datetime

import config


def get_connection(db_path=None):
    db_path = db_path or config.DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextlib.contextmanager
def transaction(conn):
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db(db_path=None):
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS candidates (
            uid TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            domain TEXT NOT NULL,
            pred_proba REAL NOT NULL,
            score REAL,
            stratum TEXT,
            batch_id INTEGER,
            source TEXT DEFAULT 'sample',

            -- URL validation
            url_status TEXT DEFAULT 'pending',
            url_http_code INTEGER,
            url_content_type TEXT,
            url_checked_at TEXT,

            -- Image download
            download_status TEXT DEFAULT 'pending',
            image_width INTEGER,
            image_height INTEGER,
            image_max_dim INTEGER,
            image_format TEXT,
            image_size_bytes INTEGER,
            local_path TEXT,
            downloaded_at TEXT,

            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            step TEXT NOT NULL,
            params TEXT,
            started_at TEXT,
            finished_at TEXT,
            items_processed INTEGER DEFAULT 0,
            items_success INTEGER DEFAULT 0,
            items_failed INTEGER DEFAULT 0,
            notes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_url_status ON candidates(url_status);
        CREATE INDEX IF NOT EXISTS idx_download_status ON candidates(download_status);
        CREATE INDEX IF NOT EXISTS idx_domain ON candidates(domain);
        CREATE INDEX IF NOT EXISTS idx_pred_proba ON candidates(pred_proba);
        CREATE INDEX IF NOT EXISTS idx_batch_id ON candidates(batch_id);
        CREATE INDEX IF NOT EXISTS idx_source ON candidates(source);
        CREATE INDEX IF NOT EXISTS idx_image_dims ON candidates(image_width, image_height);
        CREATE INDEX IF NOT EXISTS idx_image_max_dim ON candidates(image_max_dim);
    """)
    conn.close()


def start_run(conn, step, params=None):
    cur = conn.execute(
        "INSERT INTO pipeline_runs (step, params, started_at) VALUES (?, ?, ?)",
        (step, json.dumps(params) if params else None, datetime.now().isoformat())
    )
    conn.commit()
    return cur.lastrowid


def finish_run(conn, run_id, processed=0, success=0, failed=0, notes=None):
    conn.execute(
        """UPDATE pipeline_runs
           SET finished_at=?, items_processed=?, items_success=?, items_failed=?, notes=?
           WHERE id=?""",
        (datetime.now().isoformat(), processed, success, failed, notes, run_id)
    )
    conn.commit()


def insert_candidates(conn, rows, batch_id, source="sample"):
    before = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    conn.executemany(
        """INSERT OR IGNORE INTO candidates
           (uid, url, domain, pred_proba, score, stratum, batch_id, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (r["uid"], r["url"], r["domain"], r["pred_proba"],
             r.get("score"), r.get("stratum"), batch_id, source)
            for r in rows
        ]
    )
    conn.commit()

    after = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    return after - before


def get_pending(conn, step, limit=None):
    queries = {
        "url_check": "SELECT uid, url FROM candidates WHERE url_status = 'pending'",
        "download": (
            "SELECT uid, url FROM candidates "
            "WHERE url_status = 'alive' AND download_status = 'pending'"
        ),
    }
    if step not in queries:
        raise ValueError(
            f"Unknown step '{step}'. Python-side steps: {list(queries)}. "
            "VLM steps are queued by multiturn/db.py."
        )
    query = queries[step]
    if limit:
        query += f" LIMIT {limit}"
    return conn.execute(query).fetchall()


def update_url_status(conn, uid, status, http_code=None, content_type=None):
    conn.execute(
        """UPDATE candidates
           SET url_status=?, url_http_code=?, url_content_type=?, url_checked_at=?
           WHERE uid=?""",
        (status, http_code, content_type, datetime.now().isoformat(), uid)
    )


def update_download_status(conn, uid, status, width=None, height=None,
                           fmt=None, size_bytes=None, local_path=None):
    image_max_dim = max(width, height) if (width and height) else None
    conn.execute(
        """UPDATE candidates
           SET download_status=?, image_width=?, image_height=?, image_max_dim=?,
               image_format=?, image_size_bytes=?, local_path=?, downloaded_at=?
           WHERE uid=?""",
        (status, width, height, image_max_dim, fmt, size_bytes,
         str(local_path) if local_path else None,
         datetime.now().isoformat(), uid)
    )


def get_stats(conn):
    stats = {}

    stats["total"] = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]

    stats["url_alive"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE url_status = 'alive'"
    ).fetchone()[0]
    stats["url_dead"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE url_status IN ('dead', 'error')"
    ).fetchone()[0]
    stats["url_pending"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE url_status = 'pending'"
    ).fetchone()[0]

    stats["downloaded"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE download_status = 'success'"
    ).fetchone()[0]
    stats["download_failed"] = conn.execute(
        "SELECT COUNT(*) FROM candidates WHERE download_status IN ('failed', 'not_image')"
    ).fetchone()[0]
    stats["download_pending"] = conn.execute(
        "SELECT COUNT(*) FROM candidates "
        "WHERE url_status = 'alive' AND download_status = 'pending'"
    ).fetchone()[0]

    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(candidates)").fetchall()}
    if "mt_size_status" in existing_cols:
        stats["mt_size_pass"] = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE mt_size_status = 'pass'"
        ).fetchone()[0]
        stats["mt_size_fail"] = conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE mt_size_status = 'fail_too_small'"
        ).fetchone()[0]

    for step, col in [("step1", "mt_step1_status"),
                      ("step2", "mt_step2_status"),
                      ("step3", "mt_step3_status")]:
        if col in existing_cols:
            stats[f"mt_{step}_done"] = conn.execute(
                f"SELECT COUNT(*) FROM candidates WHERE {col} = 'done'"
            ).fetchone()[0]
            stats[f"mt_{step}_error"] = conn.execute(
                f"SELECT COUNT(*) FROM candidates "
                f"WHERE {col} IN ('error', 'parse_error')"
            ).fetchone()[0]

    if {"mt_step3_status", "mt_methods_count", "mt_size_status", "mt_map_language"}.issubset(existing_cols):
        stats["passes_dataset"] = conn.execute(
            "SELECT COUNT(*) FROM candidates "
            "WHERE mt_size_status='pass' "
            "  AND mt_step3_status='done' AND mt_methods_count >= 1 "
            "  AND mt_map_language='en' "
            "  AND mt_is_map=1 AND mt_is_map_dominant=1 "
            "  AND mt_is_statistical_map=1 AND mt_has_admin_units=1 "
            "  AND mt_has_quantitative_data=1"
        ).fetchone()[0]

    if {"mt_step4_passed", "mt_size_status", "mt_map_language"}.issubset(existing_cols):
        stats["passes_subset"] = conn.execute(
            "SELECT COUNT(*) FROM candidates "
            "WHERE mt_step4_passed=1 AND mt_size_status='pass' "
            "  AND mt_map_language='en' "
            "  AND mt_is_map=1 AND mt_is_map_dominant=1 "
            "  AND mt_is_statistical_map=1 AND mt_has_admin_units=1 "
            "  AND mt_has_quantitative_data=1"
        ).fetchone()[0]

    stats["batches"] = conn.execute(
        """SELECT pr.id, pr.step, pr.params, pr.started_at,
                  COUNT(c.uid) as candidates_count
           FROM pipeline_runs pr
           LEFT JOIN candidates c ON c.batch_id = pr.id
           WHERE pr.step LIKE '%sample%' OR pr.step LIKE '%add%'
           GROUP BY pr.id
           ORDER BY pr.id"""
    ).fetchall()

    stats["resolution"] = conn.execute(
        """SELECT
             COUNT(*) as count,
             MIN(image_width) as min_w, MIN(image_height) as min_h,
             MAX(image_width) as max_w, MAX(image_height) as max_h,
             AVG(image_width) as avg_w, AVG(image_height) as avg_h,
             SUM(CASE WHEN image_width < 400 OR image_height < 400 THEN 1 ELSE 0 END) as below_400,
             SUM(CASE WHEN image_width < 600 OR image_height < 600 THEN 1 ELSE 0 END) as below_600,
             SUM(CASE WHEN image_width < 800 OR image_height < 800 THEN 1 ELSE 0 END) as below_800
           FROM candidates WHERE download_status = 'success'"""
    ).fetchone()

    if "mt_has_choropleth" in existing_cols:
        stats["methods"] = conn.execute(
            """SELECT
                 SUM(CASE WHEN mt_has_choropleth = 1 THEN 1 ELSE 0 END) as choropleth,
                 SUM(CASE WHEN mt_has_diagrams = 1 THEN 1 ELSE 0 END) as diagrams,
                 SUM(CASE WHEN mt_has_isolines = 1 THEN 1 ELSE 0 END) as isolines,
                 SUM(CASE WHEN mt_has_dot_density = 1 THEN 1 ELSE 0 END) as dot_density,
                 SUM(CASE WHEN mt_has_heat_map = 1 THEN 1 ELSE 0 END) as heat_map,
                 SUM(CASE WHEN mt_has_cartogram = 1 THEN 1 ELSE 0 END) as cartogram,
                 SUM(CASE WHEN mt_has_flow_map = 1 THEN 1 ELSE 0 END) as flow_map
               FROM candidates WHERE mt_step3_status = 'done'"""
        ).fetchone()

    return stats
