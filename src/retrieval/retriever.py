from typing import List, Dict, Any, Optional
from src.config import TOP_K, COLLECTION_NAME, DB_DIR
from src.embeddings.embedder import embed_query
from src.vectordb.vector_store import get_chroma_client, get_collection, query_collection


def retrieve(
    question: str,
    k: Optional[int] = None,
    max_distance: Optional[float] = None,
    collection_name: str = COLLECTION_NAME,
    db_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-k most relevant chunks for a question from vector store."""
    n_results = k if k is not None else TOP_K
    chroma_client = get_chroma_client(db_dir=db_dir or DB_DIR)
    collection = get_collection(client=chroma_client, collection_name=collection_name)

    embedding = embed_query(question)
    chunks = query_collection(collection=collection, query_embedding=embedding, k=n_results)

    if max_distance is not None:
        chunks = [c for c in chunks if c.get("distance", 1.0) <= max_distance]

    return chunks
