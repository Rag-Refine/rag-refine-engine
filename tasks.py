import os
import logging

import requests
from celery import Celery
from groq import Groq

from converter import convert_pdf_to_markdown

logger = logging.getLogger(__name__)

celery_app = Celery(
    "rag_engine",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

_SYSTEM_PROMPT = (
    "You are a Markdown cleanup expert. Your goal is to fix table formatting, "
    "remove page artifacts (headers/footers), and ensure a clear semantic hierarchy "
    "(#, ##, ###) for RAG systems. Do not change the original meaning or omit data."
)

WEBHOOK_URL = "http://localhost:3000/api/webhooks/engine-callback"


def _send_webhook(job_id: str, markdown: str, status: str) -> None:
    try:
        requests.post(
            WEBHOOK_URL,
            json={"job_id": job_id, "markdown": markdown, "status": status},
            timeout=10,
        )
    except Exception as exc:
        logger.warning("Webhook delivery failed for job %s: %s", job_id, exc)


@celery_app.task(bind=True)
def process_pdf_task(self, tmp_path: str, file_name: str) -> None:
    job_id = self.request.id
    refined_markdown = ""

    try:
        # Step 1: Docling extraction
        with open(tmp_path, "rb") as f:
            file_bytes = f.read()
        result = convert_pdf_to_markdown(file_bytes, file_name)
        raw_markdown = result["markdown"]

        # Step 2: Groq Llama 3 refinement
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": raw_markdown},
            ],
        )
        refined_markdown = response.choices[0].message.content or raw_markdown

        _send_webhook(job_id, refined_markdown, "completed")

    except Exception as exc:
        logger.error("Task %s failed: %s", job_id, exc)
        _send_webhook(job_id, "", "failed")

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
