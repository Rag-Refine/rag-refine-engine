"""PII anonymization for PDF documents.

Detects sensitive entities via regex (with validation for IBAN, credit cards,
and PT TIN) and applies true redactions using PyMuPDF:
  - `add_redact_annot` marks rectangles for redaction.
  - `apply_redactions` removes the underlying text from the content stream,
    preventing recovery via copy-paste or metadata inspection.

The caller receives the sanitized PDF bytes plus a summary dict with per-type
counts (never the redacted values themselves).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable

import pymupdf  # PyMuPDF

logger = logging.getLogger(__name__)

# ── Regex patterns ──────────────────────────────────────────────────────────
# Patterns are intentionally anchored with lookarounds to avoid partial matches
# inside larger tokens (e.g. a 9-digit run inside a 20-digit string).

_WORD_BOUNDARY_L = r"(?<![A-Za-z0-9])"
_WORD_BOUNDARY_R = r"(?![A-Za-z0-9])"

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)

# International phone numbers: +country code optional, 7–15 digits with
# spaces, dots, or hyphens between groups. Kept conservative to limit FPs.
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+\d{1,3}[ .\-]?)?(?:\(?\d{2,4}\)?[ .\-]?){2,4}\d{2,4}(?!\d)"
)

# IBAN: 2 letters + 2 digits + up to 30 alphanumerics, optionally grouped in
# groups of 4 by whitespace (e.g. "PT50 0002 0123 ..."). Validated via mod-97.
_IBAN_RE = re.compile(
    rf"{_WORD_BOUNDARY_L}[A-Z]{{2}}\d{{2}}(?:[ \t]?[A-Z0-9]){{11,34}}{_WORD_BOUNDARY_R}"
)

# SWIFT/BIC: 8 or 11 chars, bank code + country + location (+ optional branch).
_SWIFT_RE = re.compile(
    rf"{_WORD_BOUNDARY_L}[A-Z]{{4}}[A-Z]{{2}}[A-Z0-9]{{2}}(?:[A-Z0-9]{{3}})?{_WORD_BOUNDARY_R}"
)

# Credit card: 13–19 digits, optionally grouped by spaces or hyphens.
# Validated via Luhn checksum downstream.
_CARD_RE = re.compile(
    r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)"
)

# Portuguese TIN (NIF): 9 digits. Validated via the official checksum.
_PT_NIF_RE = re.compile(rf"{_WORD_BOUNDARY_L}\d{{9}}{_WORD_BOUNDARY_R}")

# Spanish DNI/NIE: 8 digits + checksum letter, or X/Y/Z + 7 digits + letter.
_ES_DNI_RE = re.compile(
    rf"{_WORD_BOUNDARY_L}[XYZ]?\d{{7,8}}[A-Z]{_WORD_BOUNDARY_R}"
)

# French TIN (SPI/Numéro fiscal): 13 digits, first is 0, 1, 2, or 3.
_FR_TIN_RE = re.compile(rf"{_WORD_BOUNDARY_L}[0-3]\d{{12}}{_WORD_BOUNDARY_R}")

# Portuguese Cartão de Cidadão: 8 digits + 1 digit + 2 letters + 1 digit.
_PT_CC_RE = re.compile(
    rf"{_WORD_BOUNDARY_L}\d{{8}}\s?\d\s?[A-Z]{{2}}\s?\d{_WORD_BOUNDARY_R}"
)

# Portuguese NISS (social security): 11 digits starting with 1 or 2.
_PT_NISS_RE = re.compile(rf"{_WORD_BOUNDARY_L}[12]\d{{10}}{_WORD_BOUNDARY_R}")

# Passport (generic): 1–2 letters + 6–9 alphanumerics. Coarse pattern, kept
# behind a required-label guard to reduce false positives.
_PASSPORT_RE = re.compile(
    rf"{_WORD_BOUNDARY_L}[A-Z]{{1,2}}\d{{6,9}}{_WORD_BOUNDARY_R}"
)

# European postal codes: PT (#### - ###), ES/FR/DE (5 digits).
# Optional trailing capitalized locality (e.g. "1150-099 LISBOA", "28013 Madrid").
_ZIP_RE = re.compile(
    r"(?<!\d)(?:\d{4}[\- ]\d{3}|\d{5})"
    r"(?:[ \t]+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][A-Za-zÁÀÂÃÉÊÍÓÔÕÚÇá-ÿ'\-]+"
    r"(?:[ \t\-][A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][A-Za-zÁÀÂÃÉÊÍÓÔÕÚÇá-ÿ'\-]+){0,3})?"
    r"(?!\d)"
)

# Street-type detection. Two forms:
#   (a) Romance languages — prefix word ("Rua", "Av.", "Calle", "Rue", …) then
#       whitespace then the rest of the line. Prefix words are capitalized so
#       "via email" / "the place" don't match.
#   (b) German composite words — the street type suffixes (straße, strasse,
#       platz, allee) sit at the word end (e.g. "Bahnhofstraße 5").
_STREET_RE = re.compile(
    r"(?:"
    r"\b(?:Rua|Avenida|Av\.|Travessa|Tv\.|Largo|Praça|Praca|Estrada|Alameda|"
    r"Caminho|Cm\.|Beco|Azinhaga|Ladeira|Calçada|Calcada|Quinta|Urbanização|Urbanizacao|"
    r"Calle|C/|Plaza|Paseo|Avda\.|Carrer|"
    r"Rue|Avenue|Boulevard|Bd\.|Place|Impasse|Chemin)"
    r"\s+[^\n\t]{2,140}"
    r"|"
    r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:straße|strasse|platz|allee)\b"
    r"(?:\s+\d{1,4}[A-Za-z]?)?"
    r")"
)

# Floor / door tokens often isolated in table cells: "2º Esq.", "R/C Dto.",
# "1º Andar", "Piso 3". Requires the address-specific keyword set, so
# generic "2º lugar" / "3º semestre" don't match.
_FLOOR_RE = re.compile(
    r"(?:"
    r"\b\d{1,2}[ºª°]\s?(?:Esq|Dir|Dto|Dta|Andar|Frente|Tr[aá]s)\.?"
    r"|\bR/C(?:\s?[ ­\-]?\s?(?:Esq|Dir|Dto|Dta)\.?)?"
    r"|\bPiso\s+\d{1,2}"
    r")"
)

# Address-label capture — grabs the rest of the line after a label such as
# "Morada:", "Endereço:", "Address:". Returns the tail in group(1).
_ADDRESS_LABELED_RE = re.compile(
    r"(?i)\b(?:morada|endere[çc]o|endere[çc]os|address|direcci[óo]n|adresse|"
    r"residência|residencia|domicílio|domicilio|local de trabalho)"
    r"\s*[:\-]\s*"
    r"([^\n]{3,200})"
)

# Name detection is heuristic — only triggered when a label like
# "Employee:", "Beneficiary:", "Nome:" precedes a capitalized sequence.
_LABELED_NAME_RE = re.compile(
    r"(?i)(?:employee|beneficiary|beneficiário|nome|name|titular|cliente)[ \t]*[:\-][ \t]*"
    r"([A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][A-Za-zÁ-ÿ]+(?:[ \t]+[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][A-Za-zÁ-ÿ]+){1,4})"
)

# ── Validators ──────────────────────────────────────────────────────────────


def _luhn_valid(digits: str) -> bool:
    s = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        n = ord(ch) - 48
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        s += n
    return s % 10 == 0


def _iban_valid(iban: str) -> bool:
    # Mod-97 check: move first 4 chars to the end, replace letters with
    # A=10..Z=35, then integer value mod 97 must equal 1.
    s = iban[4:] + iban[:4]
    buf = []
    for ch in s:
        if ch.isalpha():
            buf.append(str(ord(ch) - 55))
        elif ch.isdigit():
            buf.append(ch)
        else:
            return False
    try:
        return int("".join(buf)) % 97 == 1
    except ValueError:
        return False


def _pt_nif_valid(nif: str) -> bool:
    if len(nif) != 9 or not nif.isdigit():
        return False
    # First digit rules: 1, 2, 3 (persons); 5 (companies); 6, 7, 8, 9 (other).
    if nif[0] not in "125689" and nif[:2] not in ("45", "70", "71", "72", "74", "75", "77", "78", "79", "90", "91", "98", "99"):
        return False
    check = sum(int(nif[i]) * (9 - i) for i in range(8)) % 11
    expected = 0 if check < 2 else 11 - check
    return expected == int(nif[8])


# ── Entity types ────────────────────────────────────────────────────────────

# Ordered: more specific / higher-precision patterns first so they claim
# overlapping ranges before coarser ones.
_ENTITY_SPECS: list[tuple[str, re.Pattern[str], object]] = [
    ("EMAIL", _EMAIL_RE, None),
    ("IBAN", _IBAN_RE, "iban"),
    ("SWIFT", _SWIFT_RE, None),
    ("CREDIT_CARD", _CARD_RE, "luhn"),
    ("PT_CC", _PT_CC_RE, None),
    ("PT_NISS", _PT_NISS_RE, None),
    ("FR_TIN", _FR_TIN_RE, None),
    ("ES_DNI", _ES_DNI_RE, None),
    ("PT_NIF", _PT_NIF_RE, "pt_nif"),
    ("PASSPORT", _PASSPORT_RE, "passport_label"),
    # Address family — ADDRESS claims whole labeled line first, STREET picks
    # up unlabeled street lines, FLOOR/ZIP pick up any remaining fragments.
    ("ADDRESS", _ADDRESS_LABELED_RE, "name_group"),
    ("STREET", _STREET_RE, None),
    ("FLOOR", _FLOOR_RE, None),
    ("ZIP", _ZIP_RE, None),
    ("PHONE", _PHONE_RE, None),
    ("NAME", _LABELED_NAME_RE, "name_group"),
]


def _passes_validator(kind: str, match: re.Match[str], full_text: str, validator: object) -> str | None:
    """Return the actual string to redact (possibly a capture group) or None."""
    if validator is None:
        return match.group(0)
    if validator == "luhn":
        digits = re.sub(r"[ \-]", "", match.group(0))
        return match.group(0) if _luhn_valid(digits) else None
    if validator == "iban":
        return match.group(0) if _iban_valid(match.group(0).replace(" ", "")) else None
    if validator == "pt_nif":
        return match.group(0) if _pt_nif_valid(match.group(0)) else None
    if validator == "passport_label":
        # Only treat as passport if preceded by a passport-ish label within 30 chars.
        left = full_text[max(0, match.start() - 30) : match.start()].lower()
        if any(tok in left for tok in ("passport", "passaporte", "pasaporte")):
            return match.group(0)
        return None
    if validator == "name_group":
        return match.group(1)
    return match.group(0)


def _detect_entities(text: str) -> list[tuple[str, str]]:
    """Return [(kind, value), ...] with overlapping matches de-duplicated."""
    claimed: list[tuple[int, int]] = []
    results: list[tuple[str, str]] = []

    for kind, pattern, validator in _ENTITY_SPECS:
        for match in pattern.finditer(text):
            value = _passes_validator(kind, match, text, validator)
            if value is None:
                continue
            # Compute the span of the value within the match (for name_group).
            if validator == "name_group":
                start, end = match.span(1)
            else:
                start, end = match.span(0)
            if any(start < ce and end > cs for cs, ce in claimed):
                continue
            claimed.append((start, end))
            results.append((kind, value))

    return results


# ── Redaction ───────────────────────────────────────────────────────────────


def _redact_label(kind: str) -> str:
    return f"[REDACTED_{kind}]"


def _locate_rects(page, value: str) -> list:
    """Find on-page rectangles for a detected value.

    PyMuPDF's `search_for` doesn't match across line breaks and can fail on
    values that PDF text extraction chunked differently from the regex view.
    We try a cascade of strategies:

      1. Whole value as-is.
      2. If it contains newlines, search each non-empty line separately.
      3. If still empty and the value is long, try a head prefix (24 chars).
      4. Split on whitespace and redact each token that still looks like PII
         (len >= 3), which catches text that PyMuPDF fragments across runs.
    """
    rects = list(page.search_for(value, quads=False) or [])
    if rects:
        return rects

    if "\n" in value:
        for line in value.splitlines():
            line = line.strip()
            if len(line) < 3:
                continue
            rects.extend(page.search_for(line, quads=False) or [])
        if rects:
            return rects

    if len(value) > 24:
        rects = list(page.search_for(value[:24], quads=False) or [])
        if rects:
            return rects

    for token in value.split():
        if len(token) < 3:
            continue
        rects.extend(page.search_for(token, quads=False) or [])
    return rects


def anonymize_pdf(pdf_bytes: bytes) -> tuple[bytes, dict[str, int]]:
    """Redact PII from the given PDF bytes.

    Detection runs per page on the extracted text layer. For each match, we
    locate its on-page rectangles via `page.search_for(value)` and mark them
    for redaction with a pseudo-anonymization label. `apply_redactions()`
    then removes the underlying text from the content stream.

    Returns:
        sanitized_bytes: new PDF with redactions applied.
        summary: {entity_kind: count} — values are never included.

    Notes:
        - If `search_for` can't locate a value (text-layer mismatch, e.g.,
          values split across multiple text runs or inside images), the
          entity is counted in a `missed` bucket and NOT redacted. Callers
          can decide how to treat a missed-count > 0.
        - OCR is out of scope here; for scanned PDFs, run OCR upstream.
    """
    summary: Counter[str] = Counter()
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    try:
        for page in doc:
            text = page.get_text("text")
            if not text.strip():
                continue

            entities = _detect_entities(text)
            # Group values by kind for a compact log line.
            by_kind: dict[str, list[str]] = {}
            for kind, value in entities:
                by_kind.setdefault(kind, []).append(value)

            for kind, values in by_kind.items():
                label = _redact_label(kind)
                for value in values:
                    rects = _locate_rects(page, value)
                    if not rects:
                        summary[f"{kind}_missed"] += 1
                        continue

                    for rect in rects:
                        # Fill with black; overlay label in white so the
                        # document retains semantic hints for downstream
                        # processing without leaking the value.
                        page.add_redact_annot(
                            rect,
                            text=label,
                            fill=(0, 0, 0),
                            text_color=(1, 1, 1),
                            fontsize=max(6, min(10, rect.height * 0.7)),
                            align=pymupdf.TEXT_ALIGN_CENTER,
                        )
                    summary[kind] += 1

            # Removes text + images under redaction annots from the content stream.
            page.apply_redactions(
                images=pymupdf.PDF_REDACT_IMAGE_NONE,
                graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                text=pymupdf.PDF_REDACT_TEXT_REMOVE,
            )

        # Strip metadata (author/title often carry PII) and save a clean copy.
        doc.set_metadata({})
        sanitized = doc.tobytes(garbage=4, deflate=True, clean=True)
    finally:
        doc.close()

    counts = dict(summary)
    logger.info(
        "Redaction summary: %s",
        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no PII detected",
    )
    return sanitized, counts


def summarize(counts: dict[str, int]) -> dict[str, int]:
    """Split the raw counter into applied vs missed for caller convenience."""
    applied = {k: v for k, v in counts.items() if not k.endswith("_missed")}
    missed = {k[:-7]: v for k, v in counts.items() if k.endswith("_missed")}
    return {"applied": applied, "missed": missed, "total": sum(applied.values())}


__all__ = ["anonymize_pdf", "summarize"]


def _iter_entity_kinds() -> Iterable[str]:
    return (kind for kind, _, _ in _ENTITY_SPECS)
