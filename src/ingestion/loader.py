import io
from pathlib import Path
from typing import List, Dict, Tuple, Any
import pypdf
from src.config import DATA_DIR, MIN_CHARS


class UploadError(Exception):
    """User-facing exception for upload parsing and validation errors."""
    pass


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
            pages.append({"text": text, "source": pdf_path.name, "file": pdf_path.name, "page": page_num})
    return pdf_files, pages


def load_pdf_bytes(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract text per page with pypdf from in-memory PDF bytes.
    
    Skips pages with fewer than 20 characters.
    Raises UploadError if the PDF is corrupt, encrypted, or contains no extractable text.
    """
    if not file_bytes:
        raise UploadError(f"The uploaded file '{filename}' is empty.")

    try:
        stream = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(stream)
    except pypdf.errors.PdfReadError:
        raise UploadError(f"The PDF file '{filename}' appears to be corrupted or invalid.") from None
    except Exception as e:
        raise UploadError(f"Failed to read '{filename}': the file may be corrupted or not a valid PDF.") from None

    if reader.is_encrypted:
        try:
            # Attempt decryption with empty password for unencrypted permissions
            decrypted = reader.decrypt("")
            if decrypted == 0:
                raise UploadError(f"The PDF file '{filename}' is password-protected or encrypted and cannot be read.")
        except Exception:
            raise UploadError(f"The PDF file '{filename}' is password-protected or encrypted and cannot be read.") from None

    pages = []
    has_any_text = False

    try:
        total_pages = len(reader.pages)
    except Exception:
        raise UploadError(f"The PDF file '{filename}' appears to be corrupted or cannot be parsed.") from None

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        clean_text = text.strip()
        if len(clean_text) >= 20:
            has_any_text = True
            pages.append({
                "file": filename,
                "source": filename,
                "page": page_num,
                "text": clean_text,
            })
        elif len(clean_text) > 0:
            has_any_text = True

    if not pages or not has_any_text:
        raise UploadError("This PDF looks scanned or image-only, so no text could be extracted.")

    return pages
