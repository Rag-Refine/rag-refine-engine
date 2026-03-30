from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from converter import convert_pdf_to_markdown


app = FastAPI(
    title="RAG-Refine Engine",
    description="PDF-to-Markdown conversion microservice optimised for LLM RAG pipelines.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConversionMetadata(BaseModel):
    page_count: int
    processing_time_seconds: float


class ConversionResponse(BaseModel):
    markdown: str
    file_name: str
    metadata: ConversionMetadata


@app.post("/convert", response_model=ConversionResponse, summary="Convert a PDF to Markdown")
async def convert_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF via multipart/form-data and returns high-fidelity Markdown
    suitable for LLM ingestion (tables, headings, and lists preserved).
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # Accept application/octet-stream as well for clients that don't set the right MIME type.
        # Strict PDF validation is handled inside the converter.
        pass

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = convert_pdf_to_markdown(file_bytes, file.filename or "document.pdf")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result
