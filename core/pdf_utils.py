"""
pdf_utils.py - Everything related to the PDF file itself (PyMuPDF / fitz):
  * extract text page by page   (used by RAG, summary, translation)
  * render a page as an image   (PDF Reader, with yellow highlights)
  * Finder: exact keyword search inside the PDF (like Ctrl+F)
"""
import re

import pymupdf as fitz  # PyMuPDF


def page_count(path) -> int:
    with fitz.open(path) as doc:
        return doc.page_count


def extract_pages(path):
    """Return a list of (page_number, text). Page numbers start at 1."""
    with fitz.open(path) as doc:
        return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]


def get_page_text(path, page_no: int) -> str:
    with fitz.open(path) as doc:
        if 1 <= page_no <= doc.page_count:
            return doc[page_no - 1].get_text("text")
    return ""


def get_pages_text(path, start: int, end: int) -> str:
    """Text of pages start..end (inclusive), each page labelled."""
    parts = []
    with fitz.open(path) as doc:
        start = max(1, start)
        end = min(doc.page_count, end)
        for p in range(start, end + 1):
            text = doc[p - 1].get_text("text").strip()
            if text:
                parts.append(f"[Page {p}]\n{text}")
    return "\n\n".join(parts)


def is_scanned(pages) -> bool:
    """True if the PDF has (almost) no selectable text - e.g. a scanned book."""
    total = sum(len(t.strip()) for _, t in pages)
    return total < 20 * max(1, len(pages))


def render_page(path, page_no: int, zoom: float = 1.5, highlight_terms=None) -> bytes:
    """Render one page to PNG bytes. Words in highlight_terms are marked yellow."""
    with fitz.open(path) as doc:
        page_no = min(max(1, page_no), doc.page_count)
        page = doc[page_no - 1]
        for term in highlight_terms or []:
            term = (term or "").strip()
            if not term:
                continue
            for rect in page.search_for(term):
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1, 0.9, 0))
                annot.update()
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), annots=True)
        return pix.tobytes("png")


def find_in_pdf(path, term: str, max_snippets: int = 3):
    """
    FINDER - exact text search (case-insensitive) in every page.
    Returns: [{"page": 4, "count": 2, "snippets": ["... text around the word ..."]}, ...]
    """
    term = term.strip()
    if not term:
        return []
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    results = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            snippets = []
            for m in matches[:max_snippets]:
                s, e = max(0, m.start() - 60), min(len(text), m.end() + 60)
                snippet = text[s:e].replace("\n", " ")
                snippets.append(("..." if s > 0 else "") + snippet + ("..." if e < len(text) else ""))
            results.append({"page": i + 1, "count": len(matches), "snippets": snippets})
    return results


# ---------------------------------------------------------------------------
# AI VISION helpers (images for multimodal AI)
# ---------------------------------------------------------------------------
def page_to_png(path, page_no: int, max_side: int = 1600) -> bytes:
    """Render a page as a clear PNG for the AI (no highlights), at most max_side pixels."""
    with fitz.open(path) as doc:
        page = doc[min(max(1, page_no), doc.page_count) - 1]
        longest = max(page.rect.width, page.rect.height) or 1
        zoom = max(1.0, min(3.0, max_side / longest))
        return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")


def image_to_png(data: bytes, max_side: int = 1600) -> bytes:
    """Convert any uploaded image (jpg/png/webp/...) to a PNG that is not too large."""
    with fitz.open(stream=data) as img_doc:
        page = img_doc[0]
        longest = max(page.rect.width, page.rect.height) or 1
        zoom = min(1.0, max_side / longest)
        return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")


def pages_without_text(path, min_chars: int = 20):
    """Page numbers that have (almost) no selectable text - scanned pages / pure images."""
    with fitz.open(path) as doc:
        return [i + 1 for i, page in enumerate(doc) if len(page.get_text("text").strip()) < min_chars]
