from typing import List, Dict, Any, Optional
import chromadb
from src.config import DB_DIR, COLLECTION_NAME, BATCH_SIZE


def get_chroma_client(db_dir: Optional[str] = None) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    path = db_dir if db_dir is not None else DB_DIR
    return chromadb.PersistentClient(path=str(path))


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
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
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
) -> List[Dict[str, Any]]:
    """Query ChromaDB with an embedding vector and return structured chunk results."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for i in range(len(results["ids"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "page": results["metadatas"][0][i]["page"],
                "distance": results["distances"][0][i],
            })
    return chunks

