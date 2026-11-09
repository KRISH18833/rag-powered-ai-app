import hashlib
import threading
import time
from typing import List, Dict, Any, Optional, Union
import chromadb
from src.config import (
    DB_DIR,
    COLLECTION_NAME,
    BATCH_SIZE,
    UPLOAD_MAX_FILES,
    UPLOAD_MAX_FILE_MB,
    UPLOAD_MAX_TOTAL_PAGES,
    UPLOAD_MAX_TOTAL_CHUNKS,
)
from src.ingestion.loader import UploadError
from src.embeddings.embedder import embed_batches


# Persistent client for course notes (read-only in query workflows)
_persistent_client: Optional[chromadb.PersistentClient] = None

# Ephemeral in-memory client shared across all sessions in the process
_ephemeral_client: Optional[chromadb.ClientAPI] = None


def get_chroma_client(db_dir: Optional[str] = None) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client for course notes."""
    global _persistent_client
    path = str(db_dir if db_dir is not None else DB_DIR)
    if _persistent_client is None or db_dir is not None:
        client = chromadb.PersistentClient(path=path)
        if db_dir is None:
            _persistent_client = client
        return client
    return _persistent_client


def get_ephemeral_client() -> chromadb.ClientAPI:
    """Return the shared in-memory ChromaDB client for session-scoped uploads."""
    global _ephemeral_client
    if _ephemeral_client is None:
        _ephemeral_client = chromadb.EphemeralClient()
    return _ephemeral_client


# Module-level tracking for upload collections LRU safeguard and per-session locks
_collection_creation_times: Dict[str, float] = {}
_collections_lock = threading.Lock()
MAX_UPLOAD_COLLECTIONS = 20

_session_locks: Dict[str, threading.Lock] = {}
_session_locks_guard = threading.Lock()


def get_session_lock(session_id: str) -> threading.Lock:
    """Get or create a per-session lock to prevent concurrent duplicate indexing."""
    with _session_locks_guard:
        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        return _session_locks[session_id]


def get_upload_collection(session_id: str) -> chromadb.Collection:
    """Get or create an isolated collection for the given session_id with LRU eviction safeguard."""
    cli = get_ephemeral_client()
    col_name = f"upload_{session_id}"
    with _collections_lock:
        if session_id not in _collection_creation_times:
            # If at or exceeding MAX_UPLOAD_COLLECTIONS, delete the oldest collections
            while len(_collection_creation_times) >= MAX_UPLOAD_COLLECTIONS:
                oldest_session = min(_collection_creation_times, key=_collection_creation_times.get)
                oldest_col_name = f"upload_{oldest_session}"
                try:
                    cli.delete_collection(oldest_col_name)
                except Exception:
                    pass
                _collection_creation_times.pop(oldest_session, None)
            _collection_creation_times[session_id] = time.time()

        return cli.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )


def reset_collection(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = COLLECTION_NAME,
    distance_metric: str = "cosine",
) -> chromadb.Collection:
    """Delete if existing and create a fresh ChromaDB collection."""
    cli = client or get_chroma_client()
    try:
        cli.delete_collection(collection_name)
    except Exception:
        pass
    return cli.create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance_metric},
    )


def get_collection(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Get an existing ChromaDB collection."""
    cli = client or get_chroma_client()
    return cli.get_collection(collection_name)


def store_chunks(
    collection: chromadb.Collection,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    batch_size: int = BATCH_SIZE,
) -> None:
    """Store chunks and their embeddings into ChromaDB collection in batches."""
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": c.get("source", c.get("file", "Unknown")),
            "file": c.get("file", c.get("source", "Unknown")),
            "page": c["page"],
        }
        for c in chunks
    ]
    documents = [c["text"] for c in chunks]

    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i: i + batch_size],
            embeddings=embeddings[i: i + batch_size],
            documents=documents[i: i + batch_size],
            metadatas=metadatas[i: i + batch_size],
        )


def query_collection(
    collection: chromadb.Collection,
    query_embedding: List[float],
    k: int = 6,
    default_source_type: str = "course",
) -> List[Dict[str, Any]]:
    """Query ChromaDB with an embedding vector and return structured chunk results."""
    # Check count to avoid query error on empty collections
    try:
        if collection.count() == 0:
            return []
    except Exception:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i] or {}
            # If meta['source'] is literally "upload", file is in meta['file']
            raw_source = meta.get("source", "")
            raw_file = meta.get("file", "")
            if raw_source == "upload":
                filename = raw_file or "uploaded_doc"
                src_type = "upload"
            else:
                filename = raw_file or raw_source or "Unknown"
                src_type = default_source_type

            chunks.append({
                "text": results["documents"][0][i],
                "source": filename,
                "file": filename,
                "page": meta.get("page", 1),
                "distance": results["distances"][0][i],
                "source_type": src_type,
            })
    return chunks


# ── Session-Scoped Upload Functions ───────────────────────────────────────────

def index_uploaded(
    session_id: str,
    filename: str,
    chunks: List[Dict[str, Any]],
    file_hash: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    progress_callback: Optional[Any] = None,
) -> Union[int, str]:
    """Validate limits, check duplicate, embed, and index uploaded document chunks in session collection.
    
    Protected by a per-session lock to avoid concurrent duplicate indexing.
    
    Returns:
        chunk_count (int) on successful index, or "duplicate" (str) if file is already indexed with same content.
    Raises:
        UploadError: if limits (files, file size, pages, chunks) are exceeded.
    """
    if not chunks:
        raise UploadError(f"No valid text chunks found in '{filename}'.")

    # 1. File size check (if bytes provided)
    if file_bytes is not None and len(file_bytes) > UPLOAD_MAX_FILE_MB * 1024 * 1024:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise UploadError(
            f"Upload limit exceeded: '{filename}' is {size_mb:.1f} MB (maximum allowed is {UPLOAD_MAX_FILE_MB} MB)."
        )

    # Compute content hash if not provided
    if not file_hash:
        if file_bytes is not None:
            file_hash = hashlib.sha256(file_bytes).hexdigest()
        else:
            joined_text = "".join(c.get("text", "") for c in chunks)
            file_hash = hashlib.sha256(joined_text.encode("utf-8")).hexdigest()

    session_lock = get_session_lock(session_id)
    with session_lock:
        col = get_upload_collection(session_id)
        existing = col.get(include=["metadatas"])
        existing_metas = existing.get("metadatas", []) or []

        # 2. Check for duplicate (same filename and same content hash)
        for m in existing_metas:
            if m.get("file") == filename and m.get("content_hash") == file_hash:
                return "duplicate"

        # If file exists with a different hash, remove the old version first
        for m in existing_metas:
            if m.get("file") == filename:
                remove_uploaded(session_id, filename)
                existing = col.get(include=["metadatas"])
                existing_metas = existing.get("metadatas", []) or []
                break

        # 3. Check session limits: max files
        current_files = set(m.get("file") for m in existing_metas if m.get("file"))
        if filename not in current_files and len(current_files) >= UPLOAD_MAX_FILES:
            raise UploadError(
                f"Upload limit exceeded: maximum {UPLOAD_MAX_FILES} files allowed per session."
            )

        # 4. Check session limits: max total pages across all files
        current_pages = set((m.get("file"), m.get("page")) for m in existing_metas if m.get("file") and m.get("page"))
        current_page_count = len(current_pages)
        remaining_pages = max(0, UPLOAD_MAX_TOTAL_PAGES - current_page_count)
        new_file_pages = set(c.get("page") for c in chunks if c.get("page"))
        new_page_count = len(new_file_pages)

        if new_page_count > remaining_pages:
            raise UploadError(
                f"Upload limit exceeded: '{filename}' has {new_page_count} pages, but only {remaining_pages} pages remain of your {UPLOAD_MAX_TOTAL_PAGES} page limit."
            )

        # 5. Check session limits: max total chunks across all files
        current_chunk_count = len(existing_metas)
        remaining_chunks = max(0, UPLOAD_MAX_TOTAL_CHUNKS - current_chunk_count)
        new_chunk_count = len(chunks)

        if new_chunk_count > remaining_chunks:
            raise UploadError(
                f"Upload limit exceeded: '{filename}' produces {new_chunk_count} chunks, but only {remaining_chunks} chunks remain of your {UPLOAD_MAX_TOTAL_CHUNKS} chunk limit."
            )

        # 6. Generate embeddings using existing embedder and callback
        embeddings = embed_batches(chunks, progress_callback=progress_callback)

        # 7. Add to session collection
        ids = [f"{filename}_{i}_{file_hash[:8]}" for i in range(len(chunks))]
        metadatas = [
            {
                "file": filename,
                "page": int(c["page"]),
                "source": "upload",
                "content_hash": file_hash,
            }
            for c in chunks
        ]
        documents = [c["text"] for c in chunks]

        col.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return len(chunks)


def list_uploaded(session_id: str) -> List[Dict[str, Any]]:
    """List uploaded files for session_id with chunk and page counts."""
    col = get_upload_collection(session_id)
    try:
        data = col.get(include=["metadatas"])
    except Exception:
        return []

    metas = data.get("metadatas", []) or []
    if not metas:
        return []

    files_map = {}
    for m in metas:
        fn = m.get("file")
        if not fn:
            continue
        if fn not in files_map:
            files_map[fn] = {"filename": fn, "chunks": 0, "pages": set()}
        files_map[fn]["chunks"] += 1
        if m.get("page") is not None:
            files_map[fn]["pages"].add(m.get("page"))

    return [
        {
            "filename": fn,
            "chunk_count": info["chunks"],
            "page_count": len(info["pages"]),
        }
        for fn, info in sorted(files_map.items())
    ]


def remove_uploaded(session_id: str, filename: str) -> None:
    """Remove all chunks associated with filename from session collection, and delete collection if empty."""
    col = get_upload_collection(session_id)
    try:
        col.delete(where={"file": filename})
    except Exception:
        pass

    # Delete session's collection when all files have been removed
    try:
        remaining = col.get(include=["metadatas"])
        if not remaining or not remaining.get("metadatas") or len(remaining["metadatas"]) == 0:
            clear_uploaded(session_id)
    except Exception:
        pass


def clear_uploaded(session_id: str) -> None:
    """Clear and delete the session collection entirely."""
    cli = get_ephemeral_client()
    col_name = f"upload_{session_id}"
    try:
        cli.delete_collection(col_name)
    except Exception:
        pass
    with _collections_lock:
        _collection_creation_times.pop(session_id, None)
    with _session_locks_guard:
        _session_locks.pop(session_id, None)
