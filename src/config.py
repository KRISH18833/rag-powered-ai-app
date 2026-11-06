import os
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


config = load_config()


def get_gemini_api_key() -> Optional[str]:
    """Resolve GEMINI_API_KEY from os.environ (Render) -> st.secrets -> .env file."""
    # 1. os.environ
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    # 2. st.secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            sec_key = st.secrets["GEMINI_API_KEY"]
            if sec_key and str(sec_key).strip():
                return str(sec_key).strip()
    except Exception:
        pass

    # 3. .env file
    load_dotenv(BASE_DIR / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    if key and key.strip():
        return key.strip()

    return None


GEMINI_API_KEY = get_gemini_api_key()

# Configuration parameters
CHAT_MODEL = config.get("chat_model", "gemini-3.8-flash")
EMBEDDING_MODEL = config.get("embedding_model", "gemini-embedding-001")
CHUNK_SIZE = int(config.get("chunk_size", 2000))
CHUNK_OVERLAP = int(config.get("overlap", config.get("chunk_overlap", 200)))
TOP_K = int(config.get("top_k", 6))
COLLECTION_NAME = config.get("collection", config.get("collection_name", "cloud_notes"))
DATA_DIR = BASE_DIR / config.get("data_dir", "data")
DB_DIR = str(BASE_DIR / config.get("db_dir", "chroma_db"))
MIN_CHARS = int(config.get("min_chars", 50))
BATCH_SIZE = int(config.get("batch_size", 50))

# Generation settings
generation_cfg = config.get("generation", {})
if not isinstance(generation_cfg, dict):
    generation_cfg = {}
TEMPERATURE = float(generation_cfg.get("temperature", config.get("temperature", 0.2)))

# Retrieval settings
retrieval_cfg = config.get("retrieval", {})
if not isinstance(retrieval_cfg, dict):
    retrieval_cfg = {}
MAX_DISTANCE = float(retrieval_cfg.get("max_distance", config.get("max_distance", 0.45)))
