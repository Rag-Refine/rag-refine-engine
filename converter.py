import time
import tempfile
import os
from pathlib import Path

from docling.document_converter import DocumentConverter


_converter = DocumentConverter()


def convert_pdf_to_markdown(file_bytes: bytes, file_name: str) -> dict:
    """
    Convert a PDF byte stream to Markdown using Docling.

    Returns a dict with keys: markdown, file_name, metadata.
    Raises ValueError for invalid/unreadable PDFs.
    Raises RuntimeError for unexpected processing failures.
    """
    start = time.perf_counter()

    suffix = Path(file_name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = _converter.convert(tmp_path)
        markdown = result.document.export_to_markdown()
        page_count = len(result.document.pages) if result.document.pages else 0
    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("invalid", "corrupt", "password", "not a pdf", "cannot open")):
            raise ValueError(f"Invalid or unreadable PDF: {exc}") from exc
        raise RuntimeError(f"Conversion failed: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    elapsed = round(time.perf_counter() - start, 3)

    return {
        "markdown": markdown,
        "file_name": file_name,
        "metadata": {
            "page_count": page_count,
            "processing_time_seconds": elapsed,
        },
    }
