import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
_DATA = ROOT / "data"
SEEDS_DIR = _DATA / "seeds"
IMAGES_DIR = _DATA / "images"
CHROMA_DIR = _DATA / "chroma"

for _d in (SEEDS_DIR, IMAGES_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DB_CONN = os.getenv(
    "DATABASE_URL",
    "postgresql://rag:ragpass@localhost:5432/tourism",
)

# Separate collections: BGE and CLIP vectors live in different spaces.
TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "images"

TEXT_MODEL = "BAAI/bge-base-en-v1.5"
IMAGE_MODEL = "clip-ViT-B-16"

CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 50

TOP_K_TEXT = 6
TOP_K_IMAGE = 4
MAX_SQL_ROWS = 10
MAX_IMAGES_TO_LLM = 3

IMAGE_SCORE_GAP = 0.03  # see pipeline.integrate

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemma-4-31b-it"
LLM_RETRIES = 5
LLM_RETRY_BASE_DELAY = 4.0

IMAGES_PER_ATTRACTION = 4
IMAGE_MAX_PX = 512
# A standard Wikimedia width, so the CDN serves a cached thumbnail instead of
# rendering one and throttling us.
IMAGE_THUMB_WIDTH = 640
IMAGE_DELAY = 1.5
IMAGE_RETRIES = 3

WIKI_API = "https://en.wikipedia.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
AMAZINGLANKA_API = "https://amazinglanka.com/wp-json/wp/v2/posts"
API_DELAY = 1.0
USER_AGENT = "SCS4203-TourismRAG/0.1 (university assignment; contact via course)"
