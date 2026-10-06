from pathlib import Path
import os

_env_file = Path(__file__).resolve().parents[2] / ".env"
if _env_file.exists():
    for line in _env_file.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

BASE_DIR = Path(__file__).parent
PREDICTIONS_FILE = BASE_DIR / "full_data" / "predictions_clean.parquet"
DB_PATH = Path(os.environ.get("STATMAP_DB", BASE_DIR / "pipeline.db"))
IMAGE_CACHE_DIR = Path(os.environ.get("STATMAP_IMAGE_CACHE", "image_cache"))

PRED_PROBA_MIN = 0.80
DOMAIN_CAP = 5
SAMPLE_TARGET = 500_000
EXCLUDE_DOMAINS = [
    "ville-data.com",
    "files.airnowtech.org",
    "droughtmonitor.unl.edu",
]
STRATA = [
    (0.70, 0.80, 0.20),
    (0.80, 0.90, 0.35),
    (0.90, 1.01, 0.45),
]

URL_TIMEOUT = 8
URL_WORKERS = 30
URL_BATCH_SIZE = 500

DOWNLOAD_TIMEOUT = 15
DOWNLOAD_WORKERS = 15
MAX_IMAGE_SIZE_MB = 20
ACCEPTED_FORMATS = {"JPEG", "PNG", "GIF", "WEBP", "BMP", "TIFF"}

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL_SONNET = "claude-sonnet-4-6"
CLAUDE_MODEL_HAIKU = "claude-haiku-4-5-20251001"
CLAUDE_MAX_IMAGE_DIM = 1024
MIN_IMAGE_DIM_PX = 400
