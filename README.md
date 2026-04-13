# RAG-Refine Engine

> PDF-to-Markdown microservice optimised for LLM RAG pipelines.

Converts any PDF into high-fidelity Markdown — preserving tables, headings, and lists — via a single HTTP endpoint. Built with [FastAPI](https://fastapi.tiangolo.com/) and [Docling](https://github.com/DS4SD/docling) (IBM Research).

---

## Prerequisites

- Python **3.10+**
- `pip`

---

## Setup

```bash
# with docker
docker compose up --build
# -----------------------------


# 1. Create the virtual environment
python3 -m venv .venv

# 2. Activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell)

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the server

### Development (auto-reload on file changes)

```bash
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
```

### Production (multiple workers)

```bash
uvicorn main:app --host 0.0.0.0 --port 8888 --workers 4
```

Once running, the following URLs are available:

| URL | Description |
|-----|-------------|
| `http://localhost:8888` | API root |
| `http://localhost:8888/docs` | Interactive docs (Swagger UI) |
| `http://localhost:8888/redoc` | Alternative docs (ReDoc) |

---

## Project structure

```
rag-refine-engine/
├── main.py           # FastAPI application, CORS config, /convert endpoint
├── converter.py      # Docling conversion logic (PDF bytes → Markdown string)
├── requirements.txt  # Pinned Python dependencies
└── README.md         # This file
```

---

## API Reference

### `POST /convert`

Convert a PDF file to Markdown.

**Request** — `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | PDF file | The document to convert |

**Response** — `200 OK` — `application/json`

```json
{
  "markdown": "# Document Title\n\n## Section\n\n| Col A | Col B |\n|-------|-------|\n| val 1 | val 2 |",
  "file_name": "report.pdf",
  "metadata": {
    "page_count": 5,
    "processing_time_seconds": 3.142
  }
}
```

**Error responses**

| Status | Meaning |
|--------|---------|
| `400` | Empty file, invalid format, or password-protected PDF |
| `500` | Internal processing failure |

---

## Testing with curl

```bash
curl -X POST http://localhost:8888/convert \
  -F "file=@/path/to/your/document.pdf"
```

Pretty-print the Markdown field only (requires `jq`):

```bash
curl -s -X POST http://localhost:8888/convert \
  -F "file=@/path/to/your/document.pdf" | jq -r '.markdown'
```

---

## CORS

The API allows cross-origin requests from `http://localhost:3000` (Next.js frontend). To change this, edit the `allow_origins` list in `main.py`.

---

## Notes

- **Cold start:** The first request after server startup is slower — Docling loads its ML models into memory at that point. Subsequent requests are significantly faster.
- **Large PDFs:** Processing time scales with page count and document complexity.
- **Password-protected PDFs:** These return a `400` error.
- **Supported input:** PDF only. Other file types may return a `400` or `500` depending on how Docling handles them.
