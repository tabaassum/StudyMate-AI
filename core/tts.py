"""
tts.py - Text to Speech (listen to a page, summary or translation).
Uses Google Text-to-Speech (gTTS). Supports English ('en') and Bangla ('bn').
Needs an internet connection.
"""
import io

from core.ai_features import has_bangla


def text_to_speech(text: str, lang: str = None, max_chars: int = 4000) -> bytes:
    """Return MP3 bytes. If lang is None, it is detected (Bangla letters -> 'bn')."""
    from gtts import gTTS

    text = (text or "").strip()
    if not text:
        raise ValueError("There is no text to read.")
    text = text[:max_chars]
    # remove simple markdown symbols so they are not read aloud
    for sym in ("**", "##", "#", "* ", "- ", "`"):
        text = text.replace(sym, " ")
    lang = lang or ("bn" if has_bangla(text) else "en")

    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    return buf.getvalue()
