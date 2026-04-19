from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402

from anonymizer import anonymize_pdf, summarize  # noqa: E402
from tasks import process_pdf_task  # noqa: E402

logger = logging.getLogger(__name__)


app = FastAPI(
    title="RAG-Refine Engine",
    description="Audit-ready PDF conversion engine: structured JSON blocks with bounding boxes and AI confidence scores.",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Redaction-Summary"],
)


class QueuedResponse(BaseModel):
    job_id: str
    status: str


@app.post("/convert", response_model=QueuedResponse, summary="Queue a PDF conversion job")
async def convert_pdf(
    file: UploadFile = File(...),
    job_id: str = Form(...),
    callback_url: str = Form(...),
):
    """
    Accepts a PDF via multipart/form-data, queues a background Celery task, and
    returns a job_id immediately. The result is delivered via webhook to the
    Next.js backend once processing completes.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Write to a persistent temp file — the Celery task owns cleanup.
    suffix = "".join(c for c in (file.filename or "doc.pdf") if c in ".abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")[-10:] or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    process_pdf_task.delay(tmp_path, file.filename or "document.pdf", job_id, callback_url)

    return QueuedResponse(job_id=job_id, status="queued")


@app.post(
    "/anonymize",
    summary="Redact PII from a PDF",
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Sanitized PDF; redaction summary in X-Redaction-Summary header.",
        }
    },
)
async def anonymize(file: UploadFile = File(...)):
    """
    Synchronously redact PII (TIN/NIF, IBAN, SWIFT, credit cards, NISS, CC,
    passports, phones, emails, ZIPs, labeled names) from a PDF and return
    the sanitized bytes. The per-type redaction summary is sent in the
    `X-Redaction-Summary` header as a JSON object.

    The original bytes are discarded as soon as the sanitized copy exists,
    and no values (only counts) are logged.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        sanitized_bytes, raw_counts = anonymize_pdf(file_bytes)
    except Exception as exc:
        logger.error("Anonymization failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Anonymization failed.") from exc
    finally:
        # Purge the original buffer reference before responding.
        del file_bytes

    summary = summarize(raw_counts)
    return Response(
        content=sanitized_bytes,
        media_type="application/pdf",
        headers={
            "X-Redaction-Summary": json.dumps(summary, separators=(",", ":")),
            "Content-Disposition": f'attachment; filename="{(file.filename or "document.pdf")}"',
        },
    )
