import re
from typing import List, Dict, Any
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_chunks(
    text: str,
    source: str,
    page_num: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks at sentence/paragraph boundaries."""
    chunks = []
    # Prefer splitting at paragraph or sentence boundaries
    boundaries = [m.end() for m in re.finditer(r'(\n\n+|\. +|\? +|! +)', text)]
    boundaries = sorted(set(boundaries))

    start = 0
    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:]
            if chunk.strip():
                chunks.append({"text": chunk, "source": source, "page": page_num})
            break

        # Find a boundary within [end - chunk_overlap, end] to avoid mid-sentence cuts
        split_at = end
        search_start = max(start + 1, end - chunk_overlap)
        candidates = [b for b in boundaries if search_start <= b <= end]
        if candidates:
            split_at = candidates[-1]

        chunk = text[start:split_at]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source, "page": page_num})

        # Step forward with overlap
        next_start = split_at - chunk_overlap
        # Align next_start to a boundary if possible
        fwd_candidates = [b for b in boundaries if next_start <= b <= split_at]
        if fwd_candidates:
            next_start = fwd_candidates[0]

        start = max(next_start, start + 1)  # always make progress

    return chunks


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Split a list of page dicts into text chunks."""
    all_chunks = []
    for p in pages:
        all_chunks.extend(
            split_chunks(
                p["text"],
                p["source"],
                p["page"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return all_chunks

