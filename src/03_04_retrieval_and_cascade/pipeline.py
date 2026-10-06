import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path

import config
import db


def smart_sample(pred_proba_min=None, domain_cap=None, sample_target=None,
                 exclude_domains=None, seed=None):
    import time
    seed = seed if seed is not None else int(time.time())
    print(f"  Random seed: {seed}")

    pred_proba_min = pred_proba_min or config.PRED_PROBA_MIN
    domain_cap = domain_cap or config.DOMAIN_CAP
    sample_target = sample_target or config.SAMPLE_TARGET
    exclude_domains = set(exclude_domains or [])

    print(f"Loading {config.PREDICTIONS_FILE}...")
    df = pd.read_parquet(config.PREDICTIONS_FILE)
    print(f"  Loaded: {len(df):,} candidates")

    df = df[df["pred_proba"] >= pred_proba_min].copy()
    print(f"  After pred_proba >= {pred_proba_min}: {len(df):,}")

    if "domain" not in df.columns:
        df["domain"] = df["url"].apply(lambda u: urlparse(u).netloc.replace("www.", ""))

    exclude_domains = exclude_domains | set(config.EXCLUDE_DOMAINS)
    if exclude_domains:
        df = df[~df["domain"].isin(exclude_domains)]
        print(f"  After excluding {len(exclude_domains)} domains: {len(df):,}")

    df = (
        df.groupby("domain", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), domain_cap), random_state=seed))
        .reset_index(drop=True)
    )
    print(f"  After domain cap ({domain_cap}): {len(df):,}")

    sampled_parts = []
    for low, high, fraction in config.STRATA:
        stratum_df = df[(df["pred_proba"] >= low) & (df["pred_proba"] < high)]
        target_n = int(sample_target * fraction)
        n = min(len(stratum_df), target_n)
        if n > 0:
            sampled = stratum_df.sample(n=n, random_state=seed)
            sampled["stratum"] = f"{low:.2f}-{high:.2f}"
            sampled_parts.append(sampled)
            print(f"  Stratum [{low:.2f}, {high:.2f}): {n:,} sampled from {len(stratum_df):,}")

    df_sample = pd.concat(sampled_parts, ignore_index=True)
    df_sample = df_sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  Total sampled: {len(df_sample):,}")

    conn = db.get_connection()
    db.init_db()

    run_id = db.start_run(conn, "sample", params={
        "pred_proba_min": pred_proba_min,
        "domain_cap": domain_cap,
        "sample_target": sample_target,
        "exclude_domains": list(exclude_domains),
    })

    rows = df_sample[["uid", "url", "domain", "pred_proba", "score", "stratum"]].to_dict("records")
    inserted = db.insert_candidates(conn, rows, batch_id=run_id, source="sample")

    db.finish_run(conn, run_id, processed=len(rows), success=inserted,
                  failed=len(rows) - inserted,
                  notes=f"{inserted} new, {len(rows) - inserted} duplicates skipped")
    conn.close()

    print(f"\n  Inserted: {inserted:,} new candidates")
    print(f"  Skipped:  {len(rows) - inserted:,} duplicates")
    return inserted, len(rows) - inserted


def _check_single_url(uid, url):
    try:
        r = requests.head(url, timeout=config.URL_TIMEOUT, allow_redirects=True)
        if r.status_code == 200:
            content_type = r.headers.get("Content-Type", "")
            return (uid, "alive", r.status_code, content_type)
        else:
            return (uid, "dead", r.status_code, None)
    except requests.exceptions.Timeout:
        return (uid, "dead", None, None)
    except requests.exceptions.ConnectionError:
        return (uid, "dead", None, None)
    except Exception:
        return (uid, "error", None, None)


def validate_urls():
    conn = db.get_connection()
    pending = db.get_pending(conn, "url_check")
    print(f"  URLs to validate: {len(pending):,}")

    if not pending:
        print("  Nothing to do.")
        conn.close()
        return

    run_id = db.start_run(conn, "url_check")
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=config.URL_WORKERS) as executor:
        futures = {
            executor.submit(_check_single_url, row["uid"], row["url"]): row["uid"]
            for row in pending
        }

        batch_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="URL validation"):
            uid, status, code, content_type = future.result()
            db.update_url_status(conn, uid, status, code, content_type)

            if status == "alive":
                success += 1
            else:
                failed += 1

            batch_count += 1
            if batch_count % config.URL_BATCH_SIZE == 0:
                conn.commit()

    conn.commit()
    db.finish_run(conn, run_id, processed=success + failed, success=success, failed=failed)
    conn.close()

    total = success + failed
    print(f"\n  Alive: {success:,} ({success/total*100:.1f}%)")
    print(f"  Dead:  {failed:,} ({failed/total*100:.1f}%)")


def _download_single_image(uid, url, image_dir):
    try:
        r = requests.get(url, timeout=config.DOWNLOAD_TIMEOUT, stream=True)
        if r.status_code != 200:
            return (uid, "failed", None, None, None, None, None)

        content = r.content
        size_bytes = len(content)

        if size_bytes > config.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            return (uid, "failed", None, None, None, size_bytes, None)

        try:
            img = Image.open(BytesIO(content))
            img.verify()
            img = Image.open(BytesIO(content))
        except Exception:
            return (uid, "not_image", None, None, None, size_bytes, None)

        width, height = img.size
        img_format = img.format

        if img_format not in config.ACCEPTED_FORMATS:
            return (uid, "not_image", width, height, img_format, size_bytes, None)

        ext = {
            "JPEG": ".jpg", "PNG": ".png", "GIF": ".gif",
            "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tiff",
        }.get(img_format, ".jpg")

        local_path = image_dir / f"{uid}{ext}"
        local_path.write_bytes(content)

        return (uid, "success", width, height, img_format, size_bytes, local_path)

    except requests.exceptions.Timeout:
        return (uid, "failed", None, None, None, None, None)
    except Exception:
        return (uid, "failed", None, None, None, None, None)


def download_images():
    image_dir = Path(config.IMAGE_CACHE_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)

    conn = db.get_connection()
    pending = db.get_pending(conn, "download")
    print(f"  Images to download: {len(pending):,}")

    if not pending:
        print("  Nothing to do.")
        conn.close()
        return

    run_id = db.start_run(conn, "download")
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as executor:
        futures = {
            executor.submit(_download_single_image, row["uid"], row["url"], image_dir): row["uid"]
            for row in pending
        }

        batch_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            uid, status, w, h, fmt, size, path = future.result()
            db.update_download_status(conn, uid, status, w, h, fmt, size, path)

            if status == "success":
                success += 1
            else:
                failed += 1

            batch_count += 1
            if batch_count % 200 == 0:
                conn.commit()

    conn.commit()
    db.finish_run(conn, run_id, processed=success + failed, success=success, failed=failed)
    conn.close()

    total = success + failed
    print(f"\n  Downloaded: {success:,} ({success/total*100:.1f}%)")
    print(f"  Failed:     {failed:,} ({failed/total*100:.1f}%)")
