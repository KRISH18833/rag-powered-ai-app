import sys
from pathlib import Path
from src.config import DATA_DIR, DB_DIR, COLLECTION_NAME, BATCH_SIZE
from src.ingestion.loader import load_pdfs
from src.chunking.chunker import split_chunks
from src.embeddings.embedder import embed_batches
from src.vectordb.vector_store import get_chroma_client, reset_collection, store_chunks


def main():
    print("=== Ingesting PDFs ===")

    # Load PDFs
    pdf_files, pages = load_pdfs(data_dir=DATA_DIR)
    print(f"Found {len(pdf_files)} PDF(s), {len(pages)} page(s) total.")

    # Chunk all pages
    all_chunks = []
    for p in pages:
        all_chunks.extend(split_chunks(p["text"], p["source"], p["page"]))
    print(f"Created {len(all_chunks)} chunk(s).")

    if not all_chunks:
        print("No chunks to embed. Exiting.")
        return

    # Embed
    print("Embedding chunks...")
    embeddings = embed_batches(all_chunks, batch_size=BATCH_SIZE)

    # Chroma setup — delete and recreate collection each run
    print("Storing in ChromaDB...")
    chroma_client = get_chroma_client(db_dir=DB_DIR)
    collection = reset_collection(client=chroma_client, collection_name=COLLECTION_NAME)

    # Upsert in batches
    store_chunks(collection=collection, chunks=all_chunks, embeddings=embeddings, batch_size=BATCH_SIZE)

    # Summary
    avg_len = sum(len(c["text"]) for c in all_chunks) / len(all_chunks)
    print("\n=== Summary ===")
    print(f"  PDFs         : {len(pdf_files)}")
    print(f"  Pages        : {len(pages)}")
    print(f"  Chunks       : {len(all_chunks)}")
    print(f"  Avg chunk len: {avg_len:.0f} chars")
    print("\n--- 3 Sample Chunks ---")
    for c in all_chunks[:3]:
        preview = c["text"][:120].replace("\n", " ")
        print(f"  [{c['source']} | p{c['page']}] {preview}...")


if __name__ == "__main__":
    main()
