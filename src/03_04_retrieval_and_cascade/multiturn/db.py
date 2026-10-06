import sqlite3
from datetime import datetime

import config as base_config


SCHEMA_MT_COLUMNS = [
    ("mt_pipeline_version", "TEXT"),

    ("mt_size_status",      "TEXT"),
    ("mt_size_min_dim",     "INTEGER"),
    ("mt_size_checked_at",  "TEXT"),

    ("mt_step1_status",          "TEXT"),
    ("mt_step1_model",           "TEXT"),
    ("mt_step1_checked_at",      "TEXT"),
    ("mt_is_map",                "INTEGER"),
    ("mt_is_map_dominant",       "INTEGER"),
    ("mt_short_description",     "TEXT"),
    ("mt_step1_raw_response",    "TEXT"),
    ("mt_step1_input_tokens",    "INTEGER"),
    ("mt_step1_output_tokens",   "INTEGER"),
    ("mt_step1_cache_read",      "INTEGER"),
    ("mt_step1_cache_creation",  "INTEGER"),
    ("mt_step1_error",           "TEXT"),

    ("mt_step2_status",          "TEXT"),
    ("mt_step2_model",           "TEXT"),
    ("mt_step2_checked_at",      "TEXT"),
    ("mt_map_language",          "TEXT"),
    ("mt_is_statistical_map",    "INTEGER"),
    ("mt_has_admin_units",       "INTEGER"),
    ("mt_data_level",            "TEXT"),
    ("mt_has_text",              "INTEGER"),
    ("mt_has_quantitative_data", "INTEGER"),
    ("mt_step2_raw_response",    "TEXT"),
    ("mt_step2_input_tokens",    "INTEGER"),
    ("mt_step2_output_tokens",   "INTEGER"),
    ("mt_step2_cache_read",      "INTEGER"),
    ("mt_step2_cache_creation",  "INTEGER"),
    ("mt_step2_error",           "TEXT"),

    ("mt_step3_status",          "TEXT"),
    ("mt_step3_model",           "TEXT"),
    ("mt_step3_checked_at",      "TEXT"),
    ("mt_has_choropleth",         "INTEGER"),
    ("mt_has_diagrams",           "INTEGER"),
    ("mt_has_dot_density",        "INTEGER"),
    ("mt_has_isolines",           "INTEGER"),
    ("mt_has_cartogram",          "INTEGER"),
    ("mt_has_flow_map",           "INTEGER"),
    ("mt_has_heat_map",           "INTEGER"),
    ("mt_methods_count",          "INTEGER"),
    ("mt_legend_is_classed",      "INTEGER"),
    ("mt_step3_confidence",       "TEXT"),
    ("mt_step3_raw_response",     "TEXT"),
    ("mt_step3_input_tokens",     "INTEGER"),
    ("mt_step3_output_tokens",    "INTEGER"),
    ("mt_step3_cache_read",       "INTEGER"),
    ("mt_step3_cache_creation",   "INTEGER"),
    ("mt_step3_error",            "TEXT"),

    ("mt_step4_status",       "TEXT"),
    ("mt_step4_checked_at",   "TEXT"),
    ("mt_step4_passed",       "INTEGER"),
    ("mt_step4_f2_passed",    "INTEGER"),
    ("mt_step4_f3_passed",    "INTEGER"),
]


def get_connection(db_path=None):
    db_path = db_path or base_config.DB_PATH
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migrate_schema_mt(db_path=None):
    conn = get_connection(db_path)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(candidates)").fetchall()}
    added = []
    for col, typ in SCHEMA_MT_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE candidates ADD COLUMN {col} {typ}")
            added.append(col)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mt_size_status  ON candidates(mt_size_status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mt_step1_status ON candidates(mt_step1_status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mt_step2_status ON candidates(mt_step2_status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mt_step3_status ON candidates(mt_step3_status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mt_step4_status ON candidates(mt_step4_status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mt_step4_passed ON candidates(mt_step4_passed)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mt_is_map       ON candidates(mt_is_map)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mt_is_stat      ON candidates(mt_is_statistical_map)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mt_data_level   ON candidates(mt_data_level)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mt_lang         ON candidates(mt_map_language)")
    conn.commit()
    conn.close()
    return added


def select_candidates_for_size_gate(conn, scope, force=False, limit=None):
    base = (
        "SELECT uid, image_width, image_height, local_path "
        "FROM candidates "
        "WHERE download_status = 'success'"
    )
    where, params = _scope_where(scope)
    sql = base + where
    if not force:
        sql += " AND mt_size_status IS NULL"
    if scope.get("pilot"):
        sql += " ORDER BY uid"
        sql += " LIMIT ?"
        params.append(int(scope["pilot"]))
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


def select_candidates_for_step1(conn, scope, force=False, limit=None):
    base = (
        "SELECT uid, image_width, image_height, local_path "
        "FROM candidates "
        "WHERE download_status = 'success' "
        "  AND mt_size_status = 'pass'"
    )
    where, params = _scope_where(scope)
    sql = base + where
    if not force:
        sql += " AND (mt_step1_status IS NULL OR mt_step1_status = 'error')"
    if scope.get("pilot"):
        sql += " ORDER BY uid"
        sql += " LIMIT ?"
        params.append(int(scope["pilot"]))
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


def select_candidates_for_step3(conn, scope, force=False, limit=None):
    base = (
        "SELECT uid, image_width, image_height, local_path, "
        "       mt_step1_raw_response, mt_step2_raw_response "
        "FROM candidates "
        "WHERE download_status = 'success' "
        "  AND mt_step2_status = 'done' "
        "  AND mt_is_map = 1 "
        "  AND mt_is_map_dominant = 1 "
        "  AND mt_is_statistical_map = 1 "
        "  AND mt_has_admin_units = 1 "
        "  AND mt_has_quantitative_data = 1 "
        "  AND mt_map_language = 'en' "
    )
    where, params = _scope_where(scope)
    sql = base + where
    if not force:
        sql += " AND (mt_step3_status IS NULL OR mt_step3_status = 'error')"
    if scope.get("pilot"):
        sql += " ORDER BY uid"
        sql += " LIMIT ?"
        params.append(int(scope["pilot"]))
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


def select_candidates_for_step2(conn, scope, force=False, limit=None):
    base = (
        "SELECT uid, image_width, image_height, local_path, mt_step1_raw_response "
        "FROM candidates "
        "WHERE download_status = 'success' "
        "  AND mt_step1_status = 'done' "
        "  AND mt_is_map = 1 "
        "  AND mt_is_map_dominant = 1"
    )
    where, params = _scope_where(scope)
    sql = base + where
    if not force:
        sql += " AND (mt_step2_status IS NULL OR mt_step2_status = 'error')"
    if scope.get("pilot"):
        sql += " ORDER BY uid"
        sql += " LIMIT ?"
        params.append(int(scope["pilot"]))
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


def _scope_where(scope):
    if scope.get("all"):
        return "", []
    if scope.get("uids"):
        uids = list(scope["uids"])
        placeholders = ",".join("?" for _ in uids)
        return f" AND uid IN ({placeholders})", uids
    if scope.get("pilot"):
        return "", []
    return "", []


def upsert_size_gate(conn, uid, status, min_dim):
    conn.execute(
        """UPDATE candidates
           SET mt_size_status     = ?,
               mt_size_min_dim    = ?,
               mt_size_checked_at = ?,
               mt_pipeline_version = COALESCE(mt_pipeline_version, ?)
           WHERE uid = ?""",
        (status, min_dim, datetime.now().isoformat(), PIPELINE_VERSION_FALLBACK(), uid),
    )


def upsert_step3_result(conn, uid, *, status, model, raw_response,
                         has_choropleth=None, has_diagrams=None,
                         has_dot_density=None,
                         has_isolines=None, has_cartogram=None,
                         has_flow_map=None, has_heat_map=None,
                         methods_count=None, legend_is_classed=None,
                         confidence=None,
                         input_tokens=0, output_tokens=0,
                         cache_read=0, cache_creation=0, error=None):
    conn.execute(
        """UPDATE candidates
           SET mt_step3_status            = ?,
               mt_step3_model             = ?,
               mt_step3_checked_at        = ?,
               mt_has_choropleth          = ?,
               mt_has_diagrams            = ?,
               mt_has_dot_density         = ?,
               mt_has_isolines            = ?,
               mt_has_cartogram           = ?,
               mt_has_flow_map            = ?,
               mt_has_heat_map            = ?,
               mt_methods_count           = ?,
               mt_legend_is_classed       = ?,
               mt_step3_confidence        = ?,
               mt_step3_raw_response      = ?,
               mt_step3_input_tokens      = ?,
               mt_step3_output_tokens     = ?,
               mt_step3_cache_read        = ?,
               mt_step3_cache_creation    = ?,
               mt_step3_error             = ?,
               mt_pipeline_version        = COALESCE(mt_pipeline_version, ?)
           WHERE uid = ?""",
        (
            status, model, datetime.now().isoformat(),
            _bool_to_int(has_choropleth), _bool_to_int(has_diagrams),
            _bool_to_int(has_dot_density),
            _bool_to_int(has_isolines), _bool_to_int(has_cartogram),
            _bool_to_int(has_flow_map), _bool_to_int(has_heat_map),
            methods_count, _bool_to_int(legend_is_classed), confidence,
            raw_response,
            input_tokens, output_tokens, cache_read, cache_creation,
            error,
            PIPELINE_VERSION_FALLBACK(),
            uid,
        ),
    )


def upsert_step2_result(conn, uid, *, status, model, raw_response,
                         map_language=None, is_statistical_map=None,
                         has_admin_units=None, data_level=None, has_text=None,
                         has_quantitative_data=None,
                         input_tokens=0, output_tokens=0,
                         cache_read=0, cache_creation=0, error=None):
    conn.execute(
        """UPDATE candidates
           SET mt_step2_status          = ?,
               mt_step2_model           = ?,
               mt_step2_checked_at      = ?,
               mt_map_language          = ?,
               mt_is_statistical_map    = ?,
               mt_has_admin_units       = ?,
               mt_data_level            = ?,
               mt_has_text              = ?,
               mt_has_quantitative_data = ?,
               mt_step2_raw_response    = ?,
               mt_step2_input_tokens    = ?,
               mt_step2_output_tokens   = ?,
               mt_step2_cache_read      = ?,
               mt_step2_cache_creation  = ?,
               mt_step2_error           = ?,
               mt_pipeline_version      = COALESCE(mt_pipeline_version, ?)
           WHERE uid = ?""",
        (
            status, model, datetime.now().isoformat(),
            map_language,
            _bool_to_int(is_statistical_map), _bool_to_int(has_admin_units),
            data_level, _bool_to_int(has_text), _bool_to_int(has_quantitative_data),
            raw_response,
            input_tokens, output_tokens, cache_read, cache_creation,
            error,
            PIPELINE_VERSION_FALLBACK(),
            uid,
        ),
    )


def stats_step2(conn):
    rows = conn.execute(
        "SELECT mt_step2_status, COUNT(*) FROM candidates "
        "WHERE download_status='success' GROUP BY mt_step2_status"
    ).fetchall()
    return {r[0] or "pending": r[1] for r in rows}


def upsert_step1_result(conn, uid, *, status, model, raw_response,
                         is_map=None, is_map_dominant=None, short_description=None,
                         input_tokens=0, output_tokens=0,
                         cache_read=0, cache_creation=0, error=None):
    conn.execute(
        """UPDATE candidates
           SET mt_step1_status         = ?,
               mt_step1_model          = ?,
               mt_step1_checked_at     = ?,
               mt_is_map               = ?,
               mt_is_map_dominant      = ?,
               mt_short_description    = ?,
               mt_step1_raw_response   = ?,
               mt_step1_input_tokens   = ?,
               mt_step1_output_tokens  = ?,
               mt_step1_cache_read     = ?,
               mt_step1_cache_creation = ?,
               mt_step1_error          = ?,
               mt_pipeline_version     = COALESCE(mt_pipeline_version, ?)
           WHERE uid = ?""",
        (
            status, model, datetime.now().isoformat(),
            _bool_to_int(is_map), _bool_to_int(is_map_dominant), short_description,
            raw_response,
            input_tokens, output_tokens, cache_read, cache_creation,
            error,
            PIPELINE_VERSION_FALLBACK(),
            uid,
        ),
    )


def upsert_step4_result(conn, uid, *, status, passed, f2_passed, f3_passed):
    conn.execute(
        """UPDATE candidates
           SET mt_step4_status      = ?,
               mt_step4_checked_at  = ?,
               mt_step4_passed      = ?,
               mt_step4_f2_passed   = ?,
               mt_step4_f3_passed   = ?,
               mt_pipeline_version  = COALESCE(mt_pipeline_version, ?)
           WHERE uid = ?""",
        (
            status, datetime.now().isoformat(),
            _bool_to_int(passed),
            _bool_to_int(f2_passed),
            _bool_to_int(f3_passed),
            PIPELINE_VERSION_FALLBACK(),
            uid,
        ),
    )


def stats_size_gate(conn):
    rows = conn.execute(
        "SELECT mt_size_status, COUNT(*) FROM candidates "
        "WHERE download_status='success' GROUP BY mt_size_status"
    ).fetchall()
    return {r[0] or "pending": r[1] for r in rows}


def stats_step1(conn):
    rows = conn.execute(
        "SELECT mt_step1_status, COUNT(*) FROM candidates "
        "WHERE download_status='success' GROUP BY mt_step1_status"
    ).fetchall()
    return {r[0] or "pending": r[1] for r in rows}


def _bool_to_int(val):
    if val is None:
        return None
    return 1 if bool(val) else 0


def PIPELINE_VERSION_FALLBACK():
    from . import config as mt_config
    return mt_config.PIPELINE_VERSION
