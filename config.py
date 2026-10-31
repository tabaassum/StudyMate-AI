"""
config.py - Project-wide settings.

Everything you may want to change (API keys, model names, folders, chunk size)
lives here or in the .env file, so the rest of the code stays clean.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"        # original PDF files
CHROMA_DIR = DATA_DIR / "chroma_db"      # vector database (RAG)
DB_PATH = DATA_DIR / "studymate.db"      # SQLite database (users, history ...)

for _folder in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# Avoid noisy warnings from HuggingFace tokenizers on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# AI providers (LLM)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()

# Ollama Cloud = https://ollama.com   |   Ollama on your own Mac = http://localhost:11434
OLLAMA_HOST = (os.getenv("OLLAMA_HOST", "").strip() or "https://ollama.com").rstrip("/")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

PROVIDERS = ["Gemini", "Groq", "Ollama", "Claude"]

DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "Gemini").strip() or "Gemini"
if DEFAULT_PROVIDER not in PROVIDERS:
    DEFAULT_PROVIDER = "Gemini"

ENV_KEYS = {
    "Gemini": GEMINI_API_KEY,
    "Claude": ANTHROPIC_API_KEY,
    "Groq": GROQ_API_KEY,
    "Ollama": OLLAMA_API_KEY,
}

KEY_LINKS = {
    "Gemini": "https://aistudio.google.com/apikey",
    "Claude": "https://console.anthropic.com/settings/keys",
    "Groq": "https://console.groq.com/keys",
    "Ollama": "https://ollama.com/settings/keys",
}


def ollama_is_local() -> bool:
    return "localhost" in OLLAMA_HOST or "127.0.0.1" in OLLAMA_HOST


def _unique(items):
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


# Fallback lists only. When an API key is set, the app asks Google / Anthropic
# for the models your key can use and shows those instead (see core/llm.py).
# First item = default model. You can also type any model name in the app.
GEMINI_MODELS = _unique([
    os.getenv("GEMINI_MODEL", "").strip(),
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
])

CLAUDE_MODELS = _unique([
    os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5").strip(),
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-opus-5-5",
])

GROQ_MODELS = _unique([
    os.getenv("GROQ_MODEL", "").strip(),
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
])

OLLAMA_MODELS = _unique([
    os.getenv("OLLAMA_MODEL", "").strip(),
    "gpt-oss:120b",
    "gpt-oss:20b",
    "gemma4:31b",
])

DEFAULT_MODELS = {
    "Gemini": GEMINI_MODELS,
    "Claude": CLAUDE_MODELS,
    "Groq": GROQ_MODELS,
    "Ollama": OLLAMA_MODELS,
}

# ---------------------------------------------------------------------------
# RAG / Vector database settings
# ---------------------------------------------------------------------------
# Multilingual model (supports English + Bangla). ~470 MB, downloaded once.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small").strip()

CHUNK_SIZE = 900        # characters per chunk
CHUNK_OVERLAP = 150     # characters shared between neighbouring chunks
TOP_K = 5               # how many chunks to retrieve for search / chatbot

# Max characters sent to the LLM in one request for summary / quiz
MAX_CONTEXT_CHARS = 60_000
