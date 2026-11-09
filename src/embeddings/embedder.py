import time
import random
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError
from src.config import get_gemini_api_key, EMBEDDING_MODEL, BATCH_SIZE
from src.ingestion.loader import UploadError


def get_embedding_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or get_gemini_api_key()
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please set it in os.environ, st.secrets, or .env.")
    return genai.Client(api_key=key)


_default_client: Optional[genai.Client] = None


def _get_client(client: Optional[genai.Client] = None) -> genai.Client:
    global _default_client
    if client is not None:
        return client
    if _default_client is None:
        _default_client = get_embedding_client()
    return _default_client


def embed_batches(
    chunks: List[Dict[str, Any]],
    model: str = EMBEDDING_MODEL,
    batch_size: int = BATCH_SIZE,
    client: Optional[genai.Client] = None,
    progress_callback: Optional[Any] = None,
) -> List[List[float]]:
    """Embed document chunks in batches with rate-limiting, retry logic, and optional progress reporting."""
    cli = _get_client(client)
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    total_batches = -(-len(texts) // batch_size) if texts else 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        batch_idx = i // batch_size + 1
        attempt = 0
        while True:
            try:
                response = cli.models.embed_content(
                    model=model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT"
                    ),
                )
                all_embeddings.extend([e.values for e in response.embeddings])
                print(f"  Embedded batch {batch_idx} / {total_batches}")
                if progress_callback:
                    try:
                        progress_callback(batch_idx, total_batches, f"Embedded batch {batch_idx} of {total_batches}")
                    except TypeError:
                        try:
                            progress_callback(batch_idx, total_batches)
                        except TypeError:
                            try:
                                progress_callback(batch_idx / total_batches)
                            except Exception:
                                pass
                    except Exception:
                        pass
                time.sleep(1)  # short sleep to respect rate limits
                break
            except APIError as e:
                code = getattr(e, "code", None)
                if code is not None and 400 <= code < 500 and code != 429:
                    raise
                attempt += 1
                if progress_callback:
                    try:
                        progress_callback(max(0, batch_idx - 1), total_batches, "Embedding service is busy, retrying...")
                    except Exception:
                        pass
                wait = min(2 ** attempt + random.uniform(0, 1), 60)
                print(f"  [RETRY {attempt}] Embedding error: {e}. Waiting {wait:.1f}s...", flush=True)
                if attempt >= 6:
                    raise UploadError("Embedding service is busy after multiple retries. Please try again in a few moments.") from e
                time.sleep(wait)
            except Exception as e:
                attempt += 1
                if progress_callback:
                    try:
                        progress_callback(max(0, batch_idx - 1), total_batches, "Embedding service is busy, retrying...")
                    except Exception:
                        pass
                wait = min(2 ** attempt + random.uniform(0, 1), 60)
                print(f"  [RETRY {attempt}] Embedding error: {e}. Waiting {wait:.1f}s...", flush=True)
                if attempt >= 6:
                    raise UploadError("Embedding service is busy after multiple retries. Please try again in a few moments.") from e
                time.sleep(wait)

    return all_embeddings


def embed_query(
    text: str,
    model: str = EMBEDDING_MODEL,
    client: Optional[genai.Client] = None,
) -> List[float]:
    """Embed a single query string for retrieval."""
    cli = _get_client(client)
    attempt = 0
    while True:
        try:
            response = cli.models.embed_content(
                model=model,
                contents=[text],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            return response.embeddings[0].values
        except APIError as e:
            code = getattr(e, "code", None)
            if code is not None and 400 <= code < 500 and code != 429:
                raise
            attempt += 1
            wait = min(2 ** attempt + random.uniform(0, 1), 60)
            print(f"  [RETRY {attempt}] Embedding error: {e}. Waiting {wait:.1f}s...")
            time.sleep(wait)
            if attempt >= 6:
                raise RuntimeError(f"Embedding failed after {attempt} retries: {e}") from e
        except Exception as e:
            attempt += 1
            wait = min(2 ** attempt + random.uniform(0, 1), 60)
            print(f"  [RETRY {attempt}] Embedding error: {e}. Waiting {wait:.1f}s...")
            time.sleep(wait)
            if attempt >= 6:
                raise RuntimeError(f"Embedding failed after {attempt} retries: {e}") from e

