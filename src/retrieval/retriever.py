from typing import List, Dict, Any, Optional
from src.config import TOP_K, COLLECTION_NAME, DB_DIR, MAX_DISTANCE, UPLOAD_MAX_DISTANCE
from src.embeddings.embedder import embed_query
from src.vectordb.vector_store import (
    get_chroma_client,
    get_collection,
    get_upload_collection,
    query_collection,
)


def retrieve(
    question: str,
    scope: str = "course",
    session_id: Optional[str] = None,
    k: Optional[int] = None,
    max_distance: Optional[float] = None,
    upload_max_distance: Optional[float] = None,
    collection_name: str = COLLECTION_NAME,
    db_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-k most relevant chunks for a question.
    
    Scopes:
      - "course": search only course notes collection (filtered by max_distance <= 0.45)
      - "upload": search only session-uploaded documents (filtered by upload_max_distance <= 0.55)
      - "both": search both using 1 query embedding, apply respective thresholds, and merge by distance.
    
    Each result carries source_type ("course" | "upload"), file, page, distance, text.
    """
    n_results = k if k is not None else TOP_K
    course_thresh = max_distance if max_distance is not None else MAX_DISTANCE
    upload_thresh = upload_max_distance if upload_max_distance is not None else UPLOAD_MAX_DISTANCE

    # Single embedding call for the question
    query_emb = embed_query(question)

    course_chunks: List[Dict[str, Any]] = []
    if scope in ("course", "both"):
        try:
            client = get_chroma_client(db_dir=db_dir or DB_DIR)
            col = get_collection(client=client, collection_name=collection_name)
            raw = query_collection(collection=col, query_embedding=query_emb, k=n_results, default_source_type="course")
            for c in raw:
                if c.get("distance", 1.0) <= course_thresh:
                    c["source_type"] = "course"
                    course_chunks.append(c)
        except Exception as e:
            print(f"[RETRIEVER] Warning querying course notes: {e}")

    upload_chunks: List[Dict[str, Any]] = []
    if scope in ("upload", "both") and session_id:
        try:
            col = get_upload_collection(session_id)
            raw = query_collection(collection=col, query_embedding=query_emb, k=n_results, default_source_type="upload")
            for c in raw:
                if c.get("distance", 1.0) <= upload_thresh:
                    c["source_type"] = "upload"
                    upload_chunks.append(c)
        except Exception as e:
            print(f"[RETRIEVER] Warning querying uploads for session {session_id}: {e}")

    # Merge and rank by distance ascending
    combined = course_chunks + upload_chunks
    combined.sort(key=lambda x: x.get("distance", 1.0))
    return combined[:n_results]
