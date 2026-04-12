import base64
import io
import time
import tempfile
import os
from pathlib import Path

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


_pipeline_options = PdfPipelineOptions()
_pipeline_options.do_table_structure = True
_pipeline_options.table_structure_options.do_cell_matching = True

_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)

_SKIPPED_LABELS = {"page_header", "page_footer"}

_LABEL_TO_TYPE = {
    "title": "heading",
    "section_header": "heading",
    "text": "paragraph",
    "table": "table",
    "picture": "figure",
    "list_item": "list_item",
    "caption": "caption",
    "footnote": "footnote",
    "formula": "formula",
    "code": "code",
}


def _label_str(item) -> str:
    """Normalise a DocItemLabel to a plain lowercase string."""
    raw = str(item.label)
    # Handles both 'DocItemLabel.TEXT' and 'text' forms
    return raw.split(".")[-1].lower()


def _normalize_bbox(bbox, page_width: float, page_height: float) -> list:
    """
    Normalize a Docling BoundingBox (bottom-left origin, points) to
    [x0, y0, x1, y1] in 0-1000 range with top-left origin.
    """
    if page_width <= 0 or page_height <= 0:
        return [0.0, 0.0, 1000.0, 1000.0]
    x0 = round((bbox.l / page_width) * 1000, 1)
    # bbox.t is the top edge in PDF coords (larger y = higher on page)
    y0 = round(((page_height - bbox.t) / page_height) * 1000, 1)
    x1 = round((bbox.r / page_width) * 1000, 1)
    y1 = round(((page_height - bbox.b) / page_height) * 1000, 1)
    return [x0, y0, x1, y1]


def _get_markdown(item, label: str, heading_level: int) -> str:
    """Return the Markdown representation of a single Docling item."""
    if label == "table":
        try:
            return item.export_to_markdown()
        except Exception:
            return getattr(item, "text", "") or ""

    text = getattr(item, "text", "") or ""

    if label in ("title", "section_header"):
        prefix = "#" * min(heading_level, 6)
        return f"{prefix} {text}"

    return text


def _group_rows_by_y(blocks: list[dict], y_tolerance: int = 10) -> list[list[dict]]:
    """
    Group blocks into visual rows based on similar top-Y coordinates.

    Blocks within `y_tolerance` units (0-1000 scale) of the first item in the
    current row are considered part of the same row. Tolerance of ~10 corresponds
    to approximately 8-10px on a standard PDF page.
    """
    if not blocks:
        return []
    sorted_blocks = sorted(blocks, key=lambda b: b["location"]["bbox"][1])
    rows: list[list[dict]] = [[sorted_blocks[0]]]
    for block in sorted_blocks[1:]:
        y = block["location"]["bbox"][1]
        row_y_min = min(b["location"]["bbox"][1] for b in rows[-1])
        if abs(y - row_y_min) <= y_tolerance:
            rows[-1].append(block)
        else:
            rows.append([block])
    return rows


def _assign_to_columns(row: list[dict]) -> list[str]:
    """
    Assign each block to one of 3 fixed columns based on its x0 (left edge).

    Column boundaries (0-1000 normalised scale):
      x0 < 200          → column 0 (left  — descriptions, company data)
      200 ≤ x0 < 400    → column 1 (centre — quantities, central data)
      x0 ≥ 400          → column 2 (right  — values, worker data)

    Returns a list of 3 strings (empty string when column has no content).
    """
    cols = ["", "", ""]
    for block in sorted(row, key=lambda b: b["location"]["bbox"][0]):
        x0 = block["location"]["bbox"][0]
        if x0 < 200:
            col_idx = 0
        elif x0 < 400:
            col_idx = 1
        else:
            col_idx = 2
        text = block["markdown"].replace("\n", " ").strip()
        if cols[col_idx]:
            cols[col_idx] += " " + text
        else:
            cols[col_idx] = text
    return cols


def _render_layout_table(table_rows: list[list[str]]) -> str:
    """
    Render rows as a 3-column invisible Markdown alignment table.

    An empty header row keeps the table visually clean while satisfying
    Markdown spec. Left column is left-aligned, centre is centred, right
    column is right-aligned (suitable for monetary values).
    """
    if not table_rows:
        return ""
    lines: list[str] = [
        "| | | |",
        "| :--- | :---: | ---: |",
    ]
    for row in table_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_spatial_markdown(blocks: list[dict]) -> str:
    """
    Convert structured blocks into layout-preserving Markdown.

    Items that share the same vertical position (within Y_TOLERANCE) are placed
    into 3-column invisible Markdown alignment tables so that horizontal layout
    is preserved (left / centre / right columns with fixed x0 thresholds).
    Sequential multi-column rows are merged into one table; single-column rows
    are emitted as plain paragraphs. Docling-generated table blocks are passed
    through unchanged and never split across columns.

    Args:
        blocks: Structured blocks as returned by convert_pdf_to_structured.

    Returns:
        A single Markdown string with spatial layout preserved via tables.
    """
    if not blocks:
        return ""

    visual_rows = _group_rows_by_y(blocks)
    output_parts: list[str] = []
    pending_table_rows: list[list[str]] = []

    def flush_pending() -> None:
        if pending_table_rows:
            output_parts.append(_render_layout_table(pending_table_rows))
            pending_table_rows.clear()

    for visual_row in visual_rows:
        # A row with a single Docling-native table → emit as-is, never split.
        if len(visual_row) == 1 and visual_row[0]["type"] == "table":
            flush_pending()
            output_parts.append(visual_row[0]["markdown"])
            continue

        # A row with a single non-table block → plain paragraph.
        if len(visual_row) == 1:
            flush_pending()
            output_parts.append(visual_row[0]["markdown"])
            continue

        # Multiple blocks on the same row → assign to fixed 3-column grid.
        pending_table_rows.append(_assign_to_columns(visual_row))

    flush_pending()
    return "\n\n".join(output_parts)


def pdf_first_page_to_base64(file_bytes: bytes, scale: float = 2.0) -> str:
    """
    Render the first page of a PDF to a PNG and return it as a base64-encoded string.

    Uses pypdfium2 for fast, dependency-light rendering. `scale=2.0` gives ~144 DPI
    on a standard A4 page — sufficient quality for vision LLMs without being excessively
    large.

    Args:
        file_bytes: Raw PDF byte content.
        scale:      Render scale factor (1.0 = 72 DPI, 2.0 = 144 DPI).

    Returns:
        Base64-encoded PNG string (no data-URI prefix).

    Raises:
        ValueError: If the PDF cannot be opened or has no pages.
    """
    try:
        doc = pdfium.PdfDocument(file_bytes)
    except Exception as exc:
        raise ValueError(f"Cannot open PDF for rendering: {exc}") from exc

    if len(doc) == 0:
        raise ValueError("PDF has no pages.")

    page = doc[0]
    bitmap = page.render(scale=scale, rotation=0)
    pil_image = bitmap.to_pil()

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def convert_pdf_to_structured(file_bytes: bytes, file_name: str) -> dict:
    """
    Convert a PDF byte stream to a list of structured content blocks.

    Each block contains:
        id, type, markdown, location {page, bbox (0-1000 top-left normalised)}
    Headings also include a `level` key (1-6).

    Returns:
        {
            "blocks": [...],
            "metadata": {"pages": int, "processing_time_seconds": float}
        }

    Raises:
        ValueError  — invalid / unreadable PDF
        RuntimeError — unexpected processing failure
    """
    start = time.perf_counter()

    suffix = Path(file_name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = _converter.convert(tmp_path)
        doc = result.document
        page_count = len(doc.pages) if doc.pages else 0

        blocks = []
        block_index = 0
        # When set, holds the tree depth of the current table so its child
        # items (rows, cells) are skipped — the whole table is emitted once.
        in_table_level: int | None = None

        for item, iter_level in doc.iterate_items():
            label = _label_str(item)

            if label in _SKIPPED_LABELS:
                continue

            # Skip items that are nested inside an already-emitted table block.
            if in_table_level is not None:
                if iter_level > in_table_level:
                    continue
                else:
                    # Back to the same or a shallower level: table scope ended.
                    in_table_level = None

            block_type = _LABEL_TO_TYPE.get(label, "paragraph")

            # Mark that the next deeper items belong to this table.
            if label == "table":
                in_table_level = iter_level

            # Prefer the semantic heading level stored on the item itself;
            # fall back to the tree depth reported by iterate_items.
            heading_level = int(getattr(item, "level", None) or iter_level or 1)

            markdown = _get_markdown(item, label, heading_level)
            if not markdown:
                continue

            block_index += 1
            block_id = f"block_{block_index:03d}"

            # Extract and normalise bounding box
            location = {"page": 1, "bbox": [0.0, 0.0, 1000.0, 1000.0]}
            if item.prov:
                prov = item.prov[0]
                page_no = prov.page_no
                page = doc.pages.get(page_no)
                if page and getattr(page, "size", None):
                    bbox_norm = _normalize_bbox(
                        prov.bbox, page.size.width, page.size.height
                    )
                    location = {"page": page_no, "bbox": bbox_norm}

            block: dict = {
                "id": block_id,
                "type": block_type,
                "markdown": markdown,
                "location": location,
            }
            if block_type == "heading":
                block["level"] = heading_level

            blocks.append(block)

    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("invalid", "corrupt", "password", "not a pdf", "cannot open")):
            raise ValueError(f"Invalid or unreadable PDF: {exc}") from exc
        raise RuntimeError(f"Conversion failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    elapsed = round(time.perf_counter() - start, 3)

    return {
        "blocks": blocks,
        "metadata": {
            "pages": page_count,
            "processing_time_seconds": elapsed,
        },
    }
