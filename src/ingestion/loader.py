from pathlib import Path
from typing import List, Dict, Tuple, Any
import pypdf
from src.config import DATA_DIR, MIN_CHARS


def load_pdfs(
    data_dir: Path = DATA_DIR,
    min_chars: int = MIN_CHARS,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """Load text and metadata from PDF files in the specified directory."""
    pages = []
    pdf_files = sorted(Path(data_dir).glob("*.pdf"))
    for pdf_path in pdf_files:
        reader = pypdf.PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if len(text) < min_chars:
                print(f"  [WARNING] {pdf_path.name} page {page_num}: only {len(text)} chars")
            pages.append({"text": text, "source": pdf_path.name, "page": page_num})
    return pdf_files, pages

