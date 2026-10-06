#!/usr/bin/env python3
"""
Download StatMapCorpus source images and verify them against the published checksums.

StatMapCorpus ships URLs, not images (see NOTICE.md). This script fetches them and tells you,
per file, whether what you got is what was annotated.

    python download_images.py --data statmapcorpus_dataset.parquet --out ./images

Verification outcomes written to the report:
    ok          sha256 matches -- byte-identical to the annotated file
    recoded     sha256 differs, pHash within --phash-tolerance -- same map, re-encoded
                by the host (annotations still apply; pixel-level work may not)
    changed     sha256 and pHash both differ -- the URL now serves different content;
                DO NOT treat the annotations as valid for this file
    failed      could not be downloaded (dead link, timeout, HTTP error, or the host
                returned something that is not an image -- e.g. a soft-404 page)

Resumable: files already present and verified are skipped, including ones previously
classified as 'recoded' (when --verify-phash is on). Downloads are atomic -- an
interrupted or failed fetch never destroys a copy you already have. Interrupt freely.

Requires: pandas, pyarrow, requests. Pillow + scipy only if you use --verify-phash
(they are imported lazily, so the script runs without them otherwise).
"""

# Copyright (c) 2026 Marta Solarz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse, hashlib, os, threading, time, pandas as pd, requests

from concurrent.futures import ThreadPoolExecutor, as_completed

UA = ("StatMapCorpus-downloader/1.0 (academic dataset reconstruction; "
      "contact: m.solarz2@uw.edu.pl)")

# Content sniffing: leading bytes for the formats the dataset actually contains.
MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"RIFF", b"BM", b"GIF8", b"II*\x00", b"MM\x00*")

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}

_local = threading.local()


def session_for_thread():
    """One Session per worker thread: requests.Session is not documented as thread-safe."""
    s = getattr(_local, "session", None)

    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept": "image/*,*/*;q=0.8"})
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _local.session = s

    return s


def sha256_of(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def phash_of(path):
    """
    Same parameters as the published `phash` column: 32x32 grayscale, 2D DCT,
    top-left 8x8 block thresholded at its median (DC term included, which is why
    bit 0 is always set), read row-major into 64 bits.
    """
    import numpy as np
    from PIL import Image
    from scipy.fftpack import dct

    im = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
    d = dct(dct(np.asarray(im, dtype=np.float64), axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:8, :8]
    val = 0

    for b in (low > np.median(low)).flatten():
        val = (val << 1) | int(b)

    return f"{val:016x}"


def hamming(a, b):
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def classify_local(path, row, tol):
    """Compare a file already on disk against the published checksums."""
    if sha256_of(path) == row["sha256"]:
        return "ok", None

    if row.get("phash"):
        try:
            d = hamming(phash_of(path), row["phash"])
            return ("recoded" if d <= tol else "changed"), f"hamming={d}"
        except Exception as e:
            return "changed", f"phash failed: {type(e).__name__}"

    return "changed", "sha256 mismatch"


def looks_like_image(path, content_type):
    """Guard against soft-404s: hosts that serve an HTML error page with HTTP 200."""
    if content_type and content_type.split(";")[0].strip().lower().startswith("text/"):
        return False

    with open(path, "rb") as f:
        head = f.read(12)

    return head.startswith(MAGIC)


def download(url, dest, timeout, retries, backoff):
    """
    Fetch to a temporary file and return its path. Never touches `dest` itself,
    so a failure here cannot destroy an existing copy.
    """
    tmp = dest + ".part"
    last = None

    for attempt in range(retries + 1):
        try:
            r = session_for_thread().get(url, timeout=timeout, stream=True)

            if r.status_code in RETRY_STATUS and attempt < retries:
                r.close()
                last = requests.HTTPError(f"HTTP {r.status_code}")
                time.sleep(backoff * (2 ** attempt))
                continue

            with r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)

            return tmp, r.headers.get("Content-Type"), None
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last = e
            retriable = not isinstance(e, requests.HTTPError) or "HTTP 5" in str(e)

            if attempt < retries and retriable:
                time.sleep(backoff * (2 ** attempt))
                continue

            break
        except Exception as e:
            last = e
            break

    if os.path.exists(tmp):
        os.remove(tmp)

    return None, None, f"{type(last).__name__}: {last}"


def fetch(row, out_dir, timeout, verify_phash, tol, retries, backoff):
    uid, url = row["uid"], row["url"]
    ext = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}.get(row.get("image_format"), ".img")
    path = os.path.join(out_dir, uid + ext)

    # Resume: trust what is already on disk if it verifies. With --verify-phash this
    # also covers files previously classified 'recoded', so they are not re-fetched
    # on every run.
    if os.path.exists(path):
        status, note = classify_local(path, row, tol if verify_phash else -1)

        if status == "ok" or (status == "recoded" and verify_phash):
            return uid, status, path, note

    tmp, ctype, err = download(url, path, timeout, retries, backoff)

    if tmp is None:
        note = err

        if os.path.exists(path):
            note += " (existing local copy kept)"

        return uid, "failed", path if os.path.exists(path) else None, note

    if not looks_like_image(tmp, ctype):
        os.remove(tmp)
        return uid, "failed", None, f"not an image (Content-Type: {ctype or 'unset'}) -- likely a soft-404"

    os.replace(tmp, path)                                  # atomic

    if not verify_phash:
        return (uid, "ok", path, None) if sha256_of(path) == row["sha256"] else (uid, "changed", path, "sha256 mismatch")

    status, note = classify_local(path, row, tol)

    return uid, status, path, note


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="statmapcorpus_dataset.parquet")
    ap.add_argument("--out", required=True, help="output directory for images")
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads (default 8; be considerate of hosts)")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retries", type=int, default=2, help="retries on timeout/connection error/5xx (default 2)")
    ap.add_argument("--backoff", type=float, default=1.0, help="base seconds for exponential backoff (default 1.0)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N records (for testing)")
    ap.add_argument("--skip-duplicates", action="store_true", help="skip byte-duplicate copies (is_duplicate_byte == True)")
    ap.add_argument("--verify-phash", action="store_true", help="on sha256 mismatch, compare pHash to tell re-encoding from a content change (needs Pillow + scipy)")
    ap.add_argument("--phash-tolerance", type=int, default=2)
    ap.add_argument("--report", default="download_report.csv")
    a = ap.parse_args()

    if a.verify_phash:                                     # fail fast, not 20k files in
        try:
            import PIL, scipy, numpy                       # noqa: F401
        except ImportError as e:
            ap.error(f"--verify-phash needs Pillow, scipy and numpy ({e})")

    df = pd.read_parquet(a.data)

    if a.skip_duplicates:
        df = df[df.is_duplicate_byte != True]

    if a.limit is not None:
        df = df.head(a.limit)

    if df.empty:
        print("nothing to download")
        return

    os.makedirs(a.out, exist_ok=True)
    print(f"to download: {len(df):,}   workers: {a.workers}")

    counts, results, t0 = {}, [], time.time()
    rows = df.to_dict("records")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, r, a.out, a.timeout, a.verify_phash,
                          a.phash_tolerance, a.retries, a.backoff): r["uid"] for r in rows}

        for i, fut in enumerate(as_completed(futs), 1):
            uid, status, path, note = fut.result()
            counts[status] = counts.get(status, 0) + 1
            results.append({"uid": uid, "status": status, "path": path, "note": note})

            if i % 200 == 0 or i == len(rows):
                el = time.time() - t0
                print(f"  {i:,}/{len(rows):,}  " +
                      "  ".join(f"{k}={v:,}" for k, v in sorted(counts.items())) +
                      f"  ({el/i:.2f}s/file, ~{el/i*(len(rows)-i)/60:.0f} min left)", flush=True)

    pd.DataFrame(results).to_csv(a.report, index=False)
    n = max(len(results), 1)

    print(f"""
--- SUMMARY ---
{'ok':<10} {counts.get('ok', 0):>7,}  ({counts.get('ok', 0)/n*100:.1f}%)  byte-identical to the annotated file
{'recoded':<10} {counts.get('recoded', 0):>7,}  same map, re-encoded by the host
{'changed':<10} {counts.get('changed', 0):>7,}  DIFFERENT CONTENT -- annotations do not apply
{'failed':<10} {counts.get('failed', 0):>7,}  could not be downloaded (link rot, soft-404)
report: {a.report}
""")

    if counts.get("changed"):
        print("WARNING: filter out 'changed' records before using the annotations.")


if __name__ == "__main__":
    main()