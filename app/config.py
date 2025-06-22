import yaml
from pathlib import Path
import sys
from app.logger import get_logger

logger = get_logger(__name__)


# 1. Determine project root (one level up from app/)
BASE_DIR = Path(__file__).resolve().parents[1]

# 2. Path to config.yaml
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

if not CONFIG_PATH.is_file():
    sys.exit(f"[config] ERROR: config file not found at {CONFIG_PATH}")

# 3. Load YAML
try:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
except Exception as e:
    sys.exit(f"[config] ERROR reading config file: {e}")

# 4. Helper to require key
def require(key):
    if key not in cfg:
        sys.exit(f"[config] ERROR: Required config key '{key}' missing in {CONFIG_PATH}")
    return cfg[key]

# 5. Extract values, resolving paths relative to BASE_DIR
#    PDF_DIR: interpret as a folder under project root (or absolute if given)
pdf_dir_raw = require("PDF_DIR")
PDF_DIR = (BASE_DIR / pdf_dir_raw).resolve() if not Path(pdf_dir_raw).is_absolute() else Path(pdf_dir_raw)

qdrant_path_raw = require("QDRANT_PATH")
# For local Qdrant embedded storage, ensure path exists or will be created under BASE_DIR
QDRANT_PATH = (BASE_DIR / qdrant_path_raw).resolve() if not Path(qdrant_path_raw).is_absolute() else Path(qdrant_path_raw)

QDRANT_COLLECTION_NAME = require("QDRANT_COLLECTION_NAME")

# Optional numeric values with defaults
CHUNK_SIZE = cfg.get("CHUNK_SIZE", 512)
CHUNK_OVERLAP = cfg.get("CHUNK_OVERLAP", 50)

EMBEDDING_MODEL_NAME = require("EMBEDDING_MODEL_NAME")
LLM_MODEL_NAME = require("LLM_MODEL_NAME")

# 6. Print summary (optional)
logger.info(f"[config] PDF_DIR = {PDF_DIR}")
logger.info(f"[config] QDRANT_PATH = {QDRANT_PATH}")
logger.info(f"[config] QDRANT_COLLECTION_NAME = {QDRANT_COLLECTION_NAME}")
logger.info(f"[config] EMBEDDING_MODEL_NAME = {EMBEDDING_MODEL_NAME}")
logger.info(f"[config] LLM_MODEL_NAME = {LLM_MODEL_NAME}")
logger.info(f"[config] CHUNK_SIZE = {CHUNK_SIZE}, CHUNK_OVERLAP = {CHUNK_OVERLAP}")
