import os
import json
import logging
import re

from dotenv import load_dotenv
import requests

load_dotenv()
from celery import Celery
from groq import Groq

from converter import convert_pdf_to_structured, format_spatial_markdown, pdf_first_page_to_base64

logger = logging.getLogger(__name__)

# No topo do ficheiro, substitui a inicialização atual por:
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

celery_app = Celery(
    "rag_engine",
    broker=REDIS_URL,
    backend=REDIS_URL # Adicionar o backend permite recuperar o estado da task se necessário
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

_AUDITOR_SYSTEM_PROMPT = (
    "You are a Data Integrity Auditor. Your task is to evaluate the Markdown conversion of a PDF element. "
    "Assign a confidence_score (0.0 to 1.0) that reflects how accurately the Markdown represents the original. "
    "If the score is < 0.8, explain WHY in the 'audit_note' field "
    "(e.g., 'Overlapping text detected', 'Broken table structure', 'Potential OCR error', 'Complex cell merging'). "
    "Return ONLY valid JSON — no prose, no code fences. "
    'Schema: {"confidence_score": <float>, "audit_note": <string|null>} '
    'Examples: {"confidence_score": 0.95, "audit_note": null} '
    '         {"confidence_score": 0.63, "audit_note": "Broken table structure — merged cells could not be resolved."}'
)

_VISION_SYSTEM_PROMPT = (
    "You will receive an image of a document page and the text extracted via OCR. "
    "Your goal is to return only pure Markdown that visually replicates the PDF layout.\n\n"
    "Rules:\n"
    "- Use Markdown tables (| --- |) to align data that sits side by side in the original "
    "(e.g. Company Data vs Employee Data, two-column layouts).\n"
    "- Convert lists of allowances and deductions into real Markdown tables with aligned columns "
    "(Description | Amount).\n"
    "- Preserve the exact numbers from the OCR text — do not invent or alter values.\n"
    "- Use headings (#, ##) to mark section boundaries visible in the image.\n"
    "- Return ONLY the Markdown — no explanations, no code fences, no commentary."
)

# Groq vision model used for layout reconstruction.
_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Block types that are sent to the Groq auditor for confidence scoring.
_AUDITED_TYPES = {"table", "formula", "figure"}

# Default confidence for non-audited block types (headings, paragraphs, etc.)
_DEFAULT_CONFIDENCE = 0.95


def _audit_block(client: Groq, block: dict) -> tuple:
    """
    Send a single block's markdown to the Groq auditor.
    Returns (confidence_score: float, audit_note: str | None).
    Falls back to (0.9, None) on any error.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _AUDITOR_SYSTEM_PROMPT},
                {"role": "user", "content": block["markdown"]},
                {"role": "system","content": _VISION_SYSTEM_PROMPT },
            ],
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "{}").strip()
        # Strip accidental markdown code fences that some models add
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        data = json.loads(raw)
        score = float(data.get("confidence_score", 0.9))
        score = max(0.0, min(1.0, score))
        note = data.get("audit_note") or None
        # Only surface audit_note when confidence is below threshold
        if score >= 0.8:
            note = None
        return score, note
    except Exception as exc:
        logger.warning("Auditor failed for block %s: %s", block["id"], exc)
        return 0.9, None


def _refine_layout_with_vision(client: Groq, image_base64: str, raw_text: str) -> str | None:
    """
    Send the first-page image and the Docling-extracted text to a Groq vision LLM
    so it can reconstruct the spatial layout as accurate Markdown tables.

    Returns the refined Markdown string, or None if the call fails (so the caller
    can fall back to the Docling-generated output).

    Args:
        client:       Initialised Groq client.
        image_base64: Base64-encoded PNG of the first PDF page.
        raw_text:     Docling-extracted Markdown (used as ground-truth for numbers).
    """
    try:
        response = client.chat.completions.create(
            model=_VISION_MODEL,
            messages=[
                {"role": "system", "content": _VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                        {
                            "type": "text",
                            "text": f"OCR Text (use these exact numbers):\n\n{raw_text}",
                        },
                    ],
                },
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        refined = (response.choices[0].message.content or "").strip()
        # Strip accidental code fences some models insert
        refined = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", refined, flags=re.DOTALL).strip()
        return refined or None
    except Exception as exc:
        logger.warning("Vision layout refinement failed (%s): %s", _VISION_MODEL, exc)
        return None


def _send_webhook(payload: dict, callback_url: str) -> None:
    job_id = payload.get("job_id", "unknown")
    status = payload.get("status", "unknown")
    block_count = len(payload.get("content", []))
    logger.info(
        "Webhook — job_id: %s | status: %s | blocks: %d",
        job_id, status, block_count,
    )
    try:
        resp = requests.post(
            callback_url,
            json=payload,
            headers={"x-webhook-secret": os.getenv("WEBHOOK_SECRET", "")},
            timeout=10,
        )
        logger.info("Webhook delivered for job %s: HTTP %s", job_id, resp.status_code)
    except Exception as exc:
        logger.error("Webhook delivery failed for job %s: %s", job_id, exc)


@celery_app.task(bind=True)
def process_pdf_task(self, tmp_path: str, file_name: str, job_id: str, callback_url: str) -> None:
    """
    Three-pass PDF processing:
      Pass 1 — Docling extracts structured blocks with bounding boxes.
      Pass 2 — Groq auditor rates conversion confidence for complex blocks.
      Pass 3 — Groq vision LLM reconstructs spatial layout from the first page image.
               Falls back to Docling spatial output if the vision API is unavailable.
    """
    try:
        # ── Pass 1: structured extraction ────────────────────────────────────
        with open(tmp_path, "rb") as f:
            file_bytes = f.read()

        result = convert_pdf_to_structured(file_bytes, file_name)
        blocks = result["blocks"]
        meta = result["metadata"]

        # ── Pass 2: Groq auditor for complex blocks ───────────────────────────
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        content = []

        for block in blocks:
            output_block: dict = {
                "id": block["id"],
                "type": block["type"],
                "markdown": block["markdown"],
                "location": block["location"],
            }

            if "level" in block:
                output_block["level"] = block["level"]

            if block["type"] in _AUDITED_TYPES:
                score, note = _audit_block(client, block)
                output_block["confidence"] = score
                if note:
                    output_block["audit_note"] = note
            else:
                output_block["confidence"] = _DEFAULT_CONFIDENCE

            content.append(output_block)

        spatial_markdown = format_spatial_markdown(blocks)

        # ── Pass 3: Vision layout refinement ─────────────────────────────────
        # Render the first PDF page to an image and ask a vision LLM to
        # reconstruct the layout as Markdown tables. Falls back to the
        # Docling spatial output if the vision API is unavailable.
        vision_markdown: str | None = None
        try:
            image_base64 = pdf_first_page_to_base64(file_bytes)
            vision_markdown = _refine_layout_with_vision(client, image_base64, spatial_markdown)
        except Exception as exc:
            logger.warning("Vision pass skipped for job %s: %s", job_id, exc)

        final_markdown = vision_markdown or spatial_markdown

        # Append the vision-generated block so the frontend can identify it.
        # Individual blocks (with bbox) are preserved in `content` for hover
        # highlighting; this extra block carries the full structured layout.
        if vision_markdown:
            content.append({
                "id": "block_structured_layout",
                "type": "structured_layout",
                "markdown": vision_markdown,
                "location": {"page": 1, "bbox": [0.0, 0.0, 1000.0, 1000.0]},
                "confidence": 0.98,
            })

        payload = {
            "job_id": job_id,
            "status": "completed",
            "metadata": {
                "pages": meta["pages"],
                "processing_time": f"{meta['processing_time_seconds']}s",
                "vision_enhanced": vision_markdown is not None,
            },
            "markdown": final_markdown,
            "content": content,
        }
        _send_webhook(payload, callback_url)

    except Exception as exc:
        logger.error("Task %s failed: %s", job_id, exc, exc_info=True)
        _send_webhook(
            {"job_id": job_id, "status": "failed", "content": []},
            callback_url,
        )

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
