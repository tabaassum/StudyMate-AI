"""
ai_features.py - All AI tasks of the app, built on top of LLMClient:
  * RAG chatbot answer        * summary (whole PDF or page range)
  * translation EN <-> BN     * explain simply
  * word meaning              * quiz (MCQ) generator
  * key topics                * Bangla query -> English (for better search)
"""
import json
import re

import config

LANG_NAMES = {"English": "English", "বাংলা": "Bangla (Bengali, written in Bangla script)"}

_BANGLA_RE = re.compile(r"[ঀ-৿]")


def has_bangla(text: str) -> bool:
    return bool(_BANGLA_RE.search(text or ""))


def _lang(language: str) -> str:
    return LANG_NAMES.get(language, language)


def _limit(text: str, max_chars: int = config.MAX_CONTEXT_CHARS) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[text truncated]"


# ---------------------------------------------------------------------------
# AI CHATBOT (RAG: Retrieval + Generation)
# ---------------------------------------------------------------------------
CHAT_SYSTEM = """You are StudyMate AI, a helpful study assistant.
Answer the student's question ONLY using the PDF context provided.
Rules:
- If the answer is not in the context, say clearly that it was not found in the uploaded PDF(s).
- Mention the page you used like (Page 4) after the relevant sentence.
- Be clear and well-structured. Use short paragraphs or bullet points.
- Write the answer in {language}."""


def answer_question(llm, question, contexts, history=None, language="English"):
    """contexts = results from rag.search()"""
    if not contexts:
        context_text = "(no relevant text was found in the PDFs)"
    else:
        context_text = "\n\n".join(
            f"--- Source {i}: {c['file_name']} (Page {c['page']}) ---\n{c['text']}"
            for i, c in enumerate(contexts, 1)
        )
    prompt = f"PDF CONTEXT:\n{context_text}\n\nQUESTION: {question}"
    return llm.generate(prompt, system=CHAT_SYSTEM.format(language=_lang(language)), history=history)


def query_for_search(llm, query: str) -> str:
    """
    If the user typed Bangla but the PDF is English, searching with an English
    translation gives better results. Falls back to the original query.
    """
    if llm is None or not has_bangla(query):
        return query
    try:
        return llm.generate(
            f"Translate this search query to English. Reply with ONLY the translation:\n{query}",
            max_tokens=200,
        ).strip().strip('"')
    except Exception:
        return query


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
STYLE_RULES = {
    "Short": "Write a short summary of about 5-7 sentences.",
    "Detailed": "Write a detailed, well-organized summary with headings for each main topic.",
    "Bullet points": "Write the summary as clear bullet points, grouped under small headings.",
}


def summarize(llm, text, style="Short", language="English"):
    """Long documents are summarized part by part, then combined (map-reduce)."""
    text = text.strip()
    if not text:
        return "No text found to summarize."
    rule = STYLE_RULES.get(style, STYLE_RULES["Short"])
    system = f"You are an expert teacher who summarizes study material. Write in {_lang(language)}."

    if len(text) <= config.MAX_CONTEXT_CHARS:
        return llm.generate(f"{rule}\n\nTEXT:\n{text}", system=system)

    # map: summarize each part
    part_size = config.MAX_CONTEXT_CHARS // 2
    parts = [text[i:i + part_size] for i in range(0, len(text), part_size)][:8]
    notes = []
    for i, part in enumerate(parts, 1):
        notes.append(llm.generate(
            f"Write concise notes (bullet points) of the key ideas in this part {i} of a document:\n\n{part}",
            system=system,
        ))
    # reduce: combine notes
    combined = "\n\n".join(f"Part {i}:\n{n}" for i, n in enumerate(notes, 1))
    return llm.generate(f"These are notes from all parts of one document. {rule}\n\nNOTES:\n{combined}",
                        system=system)


def key_topics(llm, text, language="English"):
    prompt = (f"List the 8-12 most important topics / key terms of this text, one per line, "
              f"each with a very short explanation (max 12 words). Write in {_lang(language)}.\n\n"
              f"TEXT:\n{_limit(text)}")
    return llm.generate(prompt)


# ---------------------------------------------------------------------------
# TRANSLATOR
# ---------------------------------------------------------------------------
def translate(llm, text, direction="en-bn"):
    if direction == "en-bn":
        src, tgt = "English", "Bangla (Bengali script)"
    else:
        src, tgt = "Bangla", "English"
    system = (f"You are a professional {src} to {tgt} translator. Translate naturally and accurately. "
              f"Keep technical terms understandable (you may keep the English term in brackets). "
              f"Keep the original formatting (lines, bullet points). Reply with ONLY the translation.")
    return llm.generate(_limit(text, 30_000), system=system)


# ---------------------------------------------------------------------------
# STUDY TOOLS
# ---------------------------------------------------------------------------
def explain_simply(llm, text, language="English"):
    system = (f"You are a friendly teacher. Explain the given text so that a beginner student can "
              f"understand it easily. Use simple words, a small real-life example, and short points. "
              f"Write in {_lang(language)}.")
    return llm.generate(_limit(text, 20_000), system=system)


def word_meaning(llm, word):
    prompt = f"""Give the meaning of the word/term: "{word}"
Format:
**Word:** ...
**English meaning:** ...
**বাংলা অর্থ:** ...
**Part of speech:** ...
**Example sentence:** ... (with Bangla translation)
**Synonyms:** ..."""
    return llm.generate(prompt, max_tokens=800)


def generate_quiz(llm, text, n=5, language="English"):
    """
    Returns a list of questions:
    [{"question": str, "options": [4 str], "answer": index 0-3, "explanation": str}]
    """
    prompt = f"""Create {n} multiple-choice questions from the study text below.
Each question must have exactly 4 options and only one correct answer.
Write the questions, options and explanations in {_lang(language)}.
Return ONLY valid JSON (no markdown), in this exact format:
[
  {{"question": "...", "options": ["...", "...", "...", "..."], "answer": 0, "explanation": "..."}}
]
"answer" is the index (0-3) of the correct option.

STUDY TEXT:
{_limit(text, 40_000)}"""
    raw = llm.generate(prompt, max_tokens=6000)
    return _parse_quiz(raw)


def _parse_quiz(raw: str):
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError("The AI did not return a valid quiz. Please try again.")
    data = json.loads(match.group(0))
    quiz = []
    for item in data:
        opts = [str(o) for o in item.get("options", [])][:4]
        if len(opts) < 2 or not item.get("question"):
            continue
        try:
            ans = int(item.get("answer", 0))
        except (TypeError, ValueError):
            ans = 0
        quiz.append({
            "question": str(item["question"]),
            "options": opts,
            "answer": ans if 0 <= ans < len(opts) else 0,
            "explanation": str(item.get("explanation", "")),
        })
    if not quiz:
        raise ValueError("The AI did not return any usable questions. Please try again.")
    return quiz


# ---------------------------------------------------------------------------
# AI VISION (multimodal: the AI looks at an image)
# ---------------------------------------------------------------------------
OCR_PROMPT = """Read this page image and write out ALL the text on it exactly as written
(it may be English or Bangla). Keep headings, lists and line breaks. Write tables as markdown tables.
If there is a diagram, chart or picture, add one short line like: [Figure: what it shows].
Reply with ONLY the extracted text - no introduction, no comments."""


def ocr_image(llm, image_png: bytes) -> str:
    """OCR with AI Vision: get the text from a scanned page / photo."""
    return llm.generate(OCR_PROMPT, images=[image_png], max_tokens=6000)


def vision_ask(llm, image_png: bytes, question: str, language="English") -> str:
    """Ask any question about an image (e.g. a PDF page with a diagram)."""
    system = (f"You are StudyMate AI, a helpful teacher. Look carefully at the image and answer the "
              f"student's question about it. If the answer is not visible in the image, say so. "
              f"Write in {_lang(language)}.")
    return llm.generate(question, system=system, images=[image_png])


IMAGE_TASKS = {
    "📄 Read text (OCR)": OCR_PROMPT,
    "💡 Explain this image": "Explain what this image shows and teach the main idea simply, like a friendly teacher. "
                            "Use short points and a small example.",
    "📊 Explain diagram / chart": "Explain this diagram or chart step by step: what each part means, how the parts "
                                 "connect, and the main conclusion a student should remember.",
    "✍️ Solve / answer the question": "There is a question or problem in this image. Solve it step by step and give "
                                      "the final answer clearly.",
    "🌐 Translate text → বাংলা": "Read all the text in this image and translate it into natural Bangla (Bengali script). "
                                "Reply with only the translation.",
    "🌐 Translate text → English": "Read all the text in this image and translate it into natural English. "
                                  "Reply with only the translation.",
    "📝 Make study notes": "Turn the content of this image into clear, well-organized study notes with headings "
                          "and bullet points.",
}


def image_task(llm, image_png: bytes, task: str, language="English", extra: str = "") -> str:
    prompt = IMAGE_TASKS.get(task, task)
    if not task.startswith(("📄", "🌐")):
        prompt += f"\nWrite in {_lang(language)}."
    if extra.strip():
        prompt += f"\nExtra instruction from the student: {extra.strip()}"
    return llm.generate(prompt, images=[image_png], max_tokens=6000)
