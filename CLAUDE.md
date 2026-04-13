# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prerequisites

- Python **3.10+**, pip
- **Redis** running locally (`redis-server`) — required for the Celery task broker

## Environment

```bash
# Activate venv
source .venv/bin/activate

# First-time setup: copy env template and fill in your Groq key
cp .env.example .env
# Edit .env: set GROQ_API_KEY and REDIS_URL
```

## Common commands

```bash
# Start Redis (separate terminal)
redis-server

# Start Celery worker (separate terminal) — concurrency=1 recommended (see Architecture)
celery -A tasks worker --loglevel=info --concurrency=1

# Start FastAPI dev server (separate terminal)
uvicorn main:app --host 0.0.0.0 --port 8888 --reload

# Start FastAPI production server
uvicorn main:app --host 0.0.0.0 --port 8888 --workers 4

# Test the endpoint (returns job_id immediately)
curl -X POST http://localhost:8888/convert -F "file=@/path/to/file.pdf"

# Freeze dependencies after installing new packages
pip freeze > requirements.txt
```

## Architecture

The service uses a two-process model: the FastAPI app enqueues jobs, and a separate Celery worker executes them.

### Request flow

```
Client → POST /convert → FastAPI (main.py)
           │
           └─ writes PDF bytes to NamedTemporaryFile (persistent, delete=False)
           └─ process_pdf_task.delay(tmp_path, file_name) → Redis broker
           └─ returns {"job_id": "...", "status": "queued"}

Celery worker (tasks.py)
  ├─ reads tmp_path → convert_pdf_to_markdown() [Docling]
  ├─ sends raw Markdown to Groq Llama 3 (llama3-8b-8192) for cleanup
  ├─ deletes tmp_path (finally block)
  └─ POST http://localhost:3000/api/webhooks/engine-callback
       {"job_id": "...", "markdown": "...", "status": "completed" | "failed"}
```

### Key files

- **`main.py`** — FastAPI app. Loads `.env` via `load_dotenv()` before importing `tasks` (order matters: env must be set before Celery reads `REDIS_URL`). CORS locked to `http://localhost:3000`.
- **`tasks.py`** — Celery app (`rag_engine`) + `process_pdf_task`. `Groq` client is instantiated inside the task to pick up the env var at runtime.
- **`converter.py`** — Docling `DocumentConverter` singleton (module-level). Safe because each Celery worker process imports it independently.

### Concurrency note

`--concurrency=1` is recommended for the Celery worker. Docling loads large ML models (~several GB from HuggingFace Hub, cached in `~/.cache/huggingface/`) once per worker process. Higher concurrency would load the model multiple times per machine, multiplying memory usage without throughput gains on CPU-bound extraction.

### Temp file lifecycle

The FastAPI handler writes the PDF to a `NamedTemporaryFile(delete=False)`. The Celery task is the sole owner of cleanup (via `finally: os.unlink(tmp_path)`). `converter.py` also internally creates its own temp file for Docling, which it cleans up independently.

### CORS

Allowed origin is hardcoded in `main.py`. Update `allow_origins` when adding new frontend origins.
