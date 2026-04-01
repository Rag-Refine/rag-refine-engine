from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402
import tempfile  # noqa: E402

from tasks import process_pdf_task  # noqa: E402


app = FastAPI(
    title="RAG-Refine Engine",
    description="PDF-to-Markdown conversion microservice optimised for LLM RAG pipelines.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueuedResponse(BaseModel):
    job_id: str
    status: str


@app.post("/convert", response_model=QueuedResponse, summary="Queue a PDF conversion job")
async def convert_pdf(file: UploadFile = File(...)):
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

    task = process_pdf_task.delay(tmp_path, file.filename or "document.pdf")

    return QueuedResponse(job_id=task.id, status="queued")
