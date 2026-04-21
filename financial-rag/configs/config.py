"""Central config. Change model names, chunk sizes, paths here — not in the modules."""
from pathlib import Path
import os

# --- paths ---
ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

CLEAN_TEXT_PATH = DATA_PROCESSED / "aapl_10k_clean.txt"
SECTIONS_PATH = DATA_PROCESSED / "aapl_10k_sections.json"
CHUNKS_PATH = DATA_PROCESSED / "aapl_10k_chunks.json"
INDEX_PATH = DATA_PROCESSED / "aapl_10k.faiss"
CHUNKS_META_PATH = DATA_PROCESSED / "aapl_10k_chunks_meta.json"

# --- target filing ---
TICKER = "AAPL"
FORM_TYPE = "10-K"
# sec-edgar-downloader requires an identifying user agent.
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "Financial RAG Project research@example.com")

# --- models ---
EMBED_MODEL = "text-embedding-3-small"   # 1536 dims, cheap, good enough
EMBED_DIM = 1536
GEN_MODEL = "gpt-4o-mini"                # cheap for dev; swap to gpt-4o for final answers
JUDGE_MODEL = "gpt-4o"                   # used only for eval grading

# --- chunking ---
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

# --- retrieval ---
TOP_K = 5

# --- ensure dirs exist ---
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
