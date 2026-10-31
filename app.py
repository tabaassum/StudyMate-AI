"""
app.py - StudyMate AI : Smart PDF Learning Assistant  (Streamlit user interface)

AI Engineering final project. The 4 required parts are in these files:
  1. AI Search ............ AI Search page        -> core/rag.py  (semantic search)
  2. AI Chatbot ........... AI Chatbot page       -> core/ai_features.py + core/llm.py
  3. RAG / Vector DB ...... ChromaDB              -> core/rag.py
  4. Database Connection .. SQLite                -> core/database.py

Run it:  open run.py in PyCharm and press the green ▶ Run button
   or :  streamlit run app.py
"""
import hashlib
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

import config
from core import ai_features as ai
from core import database as db
from core import pdf_utils, rag, tts
from core.llm import LLMClient, LLMError, is_vision_model, list_models, pick_vision_model

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
NAV_DASHBOARD = "🏠 Dashboard"
NAV_LIBRARY = "📁 My Library"
NAV_READER = "📖 PDF Reader"
NAV_SEARCH = "🔍 AI Search"
NAV_CHAT = "💬 AI Chatbot"
NAV_SUMMARY = "📝 Summary"
NAV_TRANSLATE = "🌐 Translator"
NAV_TOOLS = "🧠 Study Tools"
NAV_IMAGE = "📷 Image AI"
NAV_HISTORY = "🕘 History"
PAGES = [NAV_DASHBOARD, NAV_LIBRARY, NAV_READER, NAV_SEARCH, NAV_CHAT,
         NAV_SUMMARY, NAV_TRANSLATE, NAV_TOOLS, NAV_IMAGE, NAV_HISTORY]

LANGUAGES = ["English", "বাংলা"]


# ===========================================================================
# Helpers
# ===========================================================================
def uid() -> int:
    return st.session_state.user["id"]


def answer_language() -> str:
    return st.session_state.get("answer_lang", "English")


@st.cache_resource(show_spinner="Loading the AI search model (first time it downloads ~470 MB, please wait)...")
def load_embedder():
    return rag.get_model()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_models(provider, api_key):
    """Models this API key can really use (asked from the provider). [] if it fails."""
    try:
        return list_models(provider, api_key)
    except Exception:  # noqa: BLE001
        return []


def current_key(provider):
    env_key = config.ENV_KEYS.get(provider, "")
    return (st.session_state.get(f"saved_key_{provider}") or "").strip() or env_key


def needs_key(provider):
    return not (provider == "Ollama" and config.ollama_is_local())


def model_options(provider):
    """(model list, True if the list came live from the provider)."""
    key = current_key(provider)
    live = fetch_models(provider, key) if (key or not needs_key(provider)) else []
    if live:
        return live, True
    return config.DEFAULT_MODELS[provider], False


def get_llm(show_warning=True, provider=None):
    """Build the LLM client from the sidebar settings (or .env)."""
    provider = provider or st.session_state.get("provider", config.DEFAULT_PROVIDER)
    models, _ = model_options(provider)
    saved = st.session_state.get(f"saved_model_{provider}")
    model = (st.session_state.get(f"saved_custom_{provider}") or "").strip() \
        or (saved if saved in models else models[0])
    key = current_key(provider)
    try:
        return LLMClient(provider, key, model)
    except LLMError as exc:
        if show_warning:
            st.warning(f"🔑 {exc}")
        return None


VISION_SAME = "Same as main AI"


def vision_provider():
    """Which provider is used for AI Vision (images). Default: Gemini if it has a key."""
    choice = st.session_state.get("vision_provider")
    if not choice:
        choice = "Gemini" if current_key("Gemini") else VISION_SAME
    return st.session_state.get("provider", config.DEFAULT_PROVIDER) if choice == VISION_SAME else choice


def get_vision_llm(show_warning=True):
    """LLM client for AI Vision: same settings, but a model that can see images."""
    provider = vision_provider()
    llm = get_llm(show_warning, provider)
    if llm is None:
        return None
    custom = (st.session_state.get(f"saved_custom_{provider}") or "").strip()
    if not custom:
        models, _ = model_options(provider)
        llm.model = pick_vision_model(provider, models, llm.model)
    return llm


def doc_page_text(doc, page):
    """Text of one page: real PDF text, or the AI Vision (OCR) text for scanned pages."""
    text = pdf_utils.get_page_text(doc["file_path"], page).strip()
    if len(text) < 20:
        text = db.get_ocr_text(doc["id"], page) or text
    return text


def doc_pages_text(doc, start, end):
    """Text of pages start..end, using OCR text where the PDF has none."""
    ocr = db.get_ocr_pages(doc["id"])
    parts = []
    for p, text in pdf_utils.extract_pages(doc["file_path"])[start - 1:end]:
        text = text.strip()
        if len(text) < 20 and ocr.get(p):
            text = ocr[p]
        if text:
            parts.append(f"[Page {p}]\n{text}")
    return "\n\n".join(parts)


def run_ai(spinner_text, fn, *args, **kwargs):
    """Run an AI function with a spinner and show errors nicely."""
    try:
        with st.spinner(spinner_text):
            return fn(*args, **kwargs)
    except LLMError as exc:
        st.error(f"⚠️ {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"⚠️ Something went wrong: {exc}")
    return None


def _hash(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:10]


def listen_button(text, key, lang=None, label="🔊 Listen"):
    """Button that converts text to speech and shows an audio player."""
    audio_key = f"audio_{key}_{_hash(text)}"
    if st.button(label, key=f"btn_{audio_key}"):
        try:
            with st.spinner("Creating audio..."):
                st.session_state[audio_key] = tts.text_to_speech(text, lang)
        except Exception as exc:  # noqa: BLE001
            st.error(f"⚠️ Text to speech failed (internet needed): {exc}")
    if audio_key in st.session_state:
        st.audio(st.session_state[audio_key], format="audio/mp3")


def download_button(text, file_name, key, label="⬇️ Download"):
    st.download_button(label, data=text.encode("utf-8"), file_name=file_name,
                       mime="text/plain", key=key)


def doc_map():
    docs = db.get_documents(uid())
    return docs, {d["id"]: d for d in docs}


def need_documents(docs) -> bool:
    if not docs:
        st.info("📂 You have no PDFs yet. Go to **📁 My Library** and upload a PDF first.")
        st.button("Go to My Library", on_click=go_to, args=(NAV_LIBRARY,))
        return True
    return False


def doc_file_ok(doc) -> bool:
    if not Path(doc["file_path"]).exists():
        st.error("The PDF file was not found on disk. Please delete it from My Library and upload again.")
        return False
    return True


# ---------------------------------------------------------------------------
# Callbacks (run before the page is drawn, so they can safely change widgets)
# ---------------------------------------------------------------------------
def go_to(page_name):
    st.session_state.nav = page_name


def open_in_reader(doc_id, page, highlight=""):
    st.session_state.nav = NAV_READER
    st.session_state.reader_doc_id = doc_id
    st.session_state.reader_doc = doc_id
    st.session_state.reader_page = int(page)
    st.session_state.highlight_term = highlight or ""


def _reader_doc_changed():
    doc_id = st.session_state.reader_doc
    st.session_state.reader_doc_id = doc_id
    st.session_state.reader_page = db.get_progress(uid(), doc_id)
    st.session_state.highlight_term = ""


def _change_page(delta, max_page):
    p = st.session_state.get("reader_page", 1) + delta
    st.session_state.reader_page = min(max(1, p), max_page)


def _set_page(page):
    st.session_state.reader_page = int(page)


def _jump_page():
    st.session_state.reader_page = int(st.session_state.page_jump)


def _clear_highlight():
    st.session_state.highlight_term = ""


def _set_feedback(msg_index, chat_id, value):
    db.set_feedback(chat_id, uid(), value)
    st.session_state.chat_messages[msg_index]["feedback"] = value


def _ask(question):
    st.session_state.pending_q = question


def _logout():
    for k in list(st.session_state.keys()):
        del st.session_state[k]


# ===========================================================================
# Login / Register
# ===========================================================================
def auth_page():
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("📚 StudyMate AI")
        st.caption("Smart PDF Learning Assistant — Upload · Read · Search · Chat · Summarize · Translate")
        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login", type="primary", width="stretch"):
                    user = db.verify_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Wrong username or password.")

        with tab_register:
            with st.form("register_form"):
                username = st.text_input("Username ")
                email = st.text_input("Email (optional)")
                password = st.text_input("Password (min 6 characters)", type="password")
                confirm = st.text_input("Confirm password", type="password")
                if st.form_submit_button("Create account", type="primary", width="stretch"):
                    if len(username.strip()) < 3:
                        st.error("Username must be at least 3 characters.")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif password != confirm:
                        st.error("Passwords do not match.")
                    else:
                        ok, result = db.create_user(username, email, password)
                        if ok:
                            st.success("✅ Account created! Now log in from the Login tab.")
                        else:
                            st.error(result)


# ===========================================================================
# Sidebar: menu + AI settings
# ===========================================================================
def _persist(widget_key, saved_key, default):
    """Keep a sidebar value even when its widget is hidden (Streamlit forgets hidden widgets)."""
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(saved_key, default)


def sidebar():
    with st.sidebar:
        st.markdown("## 📚 StudyMate AI")
        st.caption(f"👤 Logged in as **{st.session_state.user['username']}**")

        if "nav" not in st.session_state:
            st.session_state.nav = NAV_DASHBOARD
        st.radio("Menu", PAGES, key="nav", label_visibility="collapsed")

        st.divider()
        with st.expander("⚙️ AI Settings", expanded=get_llm(show_warning=False) is None):
            _persist("w_provider", "provider", config.DEFAULT_PROVIDER)
            provider = st.radio("AI provider", config.PROVIDERS, key="w_provider", horizontal=True,
                                help="Free: Gemini, Groq, Ollama. Paid: Claude. "
                                     "If one says 'rate limit', just switch to another.")
            st.session_state.provider = provider
            if provider == "Ollama":
                st.caption("☁️ Ollama Cloud" if not config.ollama_is_local()
                           else "💻 Ollama on this computer (no key needed)")

            models, live = model_options(provider)
            wkey = f"w_model_{provider}"
            chosen = st.session_state.get(wkey, st.session_state.get(f"saved_model_{provider}"))
            st.session_state[wkey] = chosen if chosen in models else models[0]
            st.session_state[f"saved_model_{provider}"] = st.selectbox("Model", models, key=wkey)
            if live:
                st.caption(f"✅ {len(models)} models available for your key (best one selected first)")
            elif current_key(provider) or not needs_key(provider):
                st.caption("⚠️ Could not load the model list — using the default list.")
            if st.button("🔄 Refresh model list", key=f"refresh_models_{provider}"):
                fetch_models.clear()
                st.rerun()

            _persist(f"w_custom_{provider}", f"saved_custom_{provider}", "")
            st.session_state[f"saved_custom_{provider}"] = st.text_input(
                "Custom model name (optional)", key=f"w_custom_{provider}",
                help="Leave empty to use the model above. Type a name here to use any other model.")

            env_key = config.ENV_KEYS.get(provider, "")
            _persist(f"w_key_{provider}", f"saved_key_{provider}", "")
            if needs_key(provider):
                st.session_state[f"saved_key_{provider}"] = st.text_input(
                    f"{provider} API key", type="password", key=f"w_key_{provider}",
                    placeholder="Loaded from .env ✓" if env_key else "Paste your API key here")

            llm = get_llm(show_warning=False)
            if llm:
                st.success(f"Ready: {llm.label}")
            else:
                st.warning(f"Add a {provider} API key. Get one here: {config.KEY_LINKS[provider]}")

            st.markdown("---")
            vision_options = [VISION_SAME] + config.PROVIDERS
            _persist("w_vision_provider", "vision_provider",
                     "Gemini" if current_key("Gemini") else VISION_SAME)
            st.session_state.vision_provider = st.selectbox(
                "👁️ Vision AI (for images & scanned pages)", vision_options, key="w_vision_provider",
                help="AI Vision needs a model that can see images. Gemini (free) works best. "
                     "Most Groq / Ollama models can only read text.")
            vllm = get_vision_llm(show_warning=False)
            if vllm and is_vision_model(vllm.provider, vllm.model):
                st.caption(f"👁️ Vision ready: {vllm.label}")
            elif vllm:
                st.caption(f"⚠️ {vllm.label} may not read images — Gemini is recommended for vision.")
            else:
                st.caption(f"⚠️ Add a {vision_provider()} API key to use AI Vision.")

        _persist("w_answer_lang", "answer_lang", "English")
        st.session_state.answer_lang = st.radio("🗣️ AI answer language", LANGUAGES,
                                                key="w_answer_lang", horizontal=True)

        st.divider()
        st.caption("🌙 Dark mode: ⋮ menu (top right) → Settings → Theme")
        st.button("🚪 Logout", on_click=_logout, width="stretch")


# ===========================================================================
# 🏠 Dashboard
# ===========================================================================
def dashboard_page():
    st.title(f"🏠 Welcome, {st.session_state.user['username']}!")
    stats = db.get_stats(uid())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 PDFs", stats["documents"])
    c2.metric("📃 Total pages", stats["pages"])
    c3.metric("💬 Questions asked", stats["questions"])
    c4.metric("🏆 Quiz average", f"{stats['quiz_avg']}%" if stats["quiz_avg"] is not None else "—")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🔍 Searches", stats["searches"])
    c6.metric("🌐 Translations", stats["translations"])
    c7.metric("🧪 Quizzes taken", stats["quizzes"])
    c8.metric("👁️ Vision tasks", stats["vision"])

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("📖 Continue reading")
        recent = db.get_recent_documents(uid(), 5)
        if not recent:
            st.info("No PDFs yet. Upload one in **📁 My Library**.")
            st.button("Upload a PDF", on_click=go_to, args=(NAV_LIBRARY,), type="primary")
        for d in recent:
            last = d.get("last_page") or 1
            total = max(1, d["page_count"])
            with st.container(border=True):
                a, b = st.columns([4, 1], vertical_alignment="center")
                a.markdown(f"**{d['file_name']}**")
                a.progress(min(1.0, last / total), text=f"Page {last} of {total}")
                b.button("Open", key=f"dash_open_{d['id']}", on_click=open_in_reader, args=(d["id"], last))

    with right:
        st.subheader("💬 Recent questions")
        chats = db.get_chats(uid(), 5)
        if not chats:
            st.info("No questions yet. Try the **💬 AI Chatbot**.")
        for c in chats:
            with st.container(border=True):
                st.markdown(f"**Q:** {c['question']}")
                st.caption(f"{c['created_at']} · {c['provider'] or ''}")

    with st.expander("ℹ️ How does StudyMate AI work? (project architecture)"):
        st.markdown("""
| Part | What it does | Technology |
|---|---|---|
| **Database Connection** | Users, upload history, chat/search history, notes, bookmarks, quiz scores | SQLite (`core/database.py`) |
| **RAG / Vector DB** | PDF → text → chunks → embeddings → stored in a vector database | ChromaDB + sentence-transformers (`core/rag.py`) |
| **AI Search** | Finds text by *meaning*, not exact words (semantic search) | Vector similarity (cosine) |
| **AI Chatbot** | Retrieves the best chunks, then the LLM writes an answer with page numbers | Gemini / Groq / Ollama / Claude (`core/llm.py`) |
| **AI Vision (multimodal)** | Reads scanned pages (OCR), explains diagrams, answers questions about images | Gemini / Claude vision models |

**Flow:** Upload PDF → extract text (PyMuPDF) → split into chunks → embed → ChromaDB
→ question is embedded → top-k similar chunks retrieved → sent to Gemini/Groq/Ollama/Claude → answer + source pages → saved in SQLite.
""")


# ===========================================================================
# 📁 My Library (upload + upload history)
# ===========================================================================
def process_uploads(files):
    load_embedder()
    user_dir = config.UPLOAD_DIR / str(uid())
    user_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        safe_name = re.sub(r"[^\w.\- ]", "_", f.name)
        path = user_dir / f"{datetime.now():%Y%m%d%H%M%S}_{safe_name}"
        path.write_bytes(f.getvalue())
        try:
            pages = pdf_utils.extract_pages(path)
        except Exception as exc:  # noqa: BLE001
            path.unlink(missing_ok=True)
            st.error(f"❌ {f.name}: could not read this PDF ({exc}).")
            continue

        doc_id = db.add_document(uid(), f.name, str(path), len(pages))
        if pdf_utils.is_scanned(pages):
            st.warning(f"📷 **{f.name}** looks like a scanned PDF (images, no text). "
                       "Use **👁️ Read scanned pages with AI Vision** below so AI Search and the Chatbot can use it.")

        bar = st.progress(0.0, text=f"🧠 Creating embeddings for {f.name} ...")
        try:
            n = rag.index_document(doc_id, uid(), f.name, pages,
                                   progress_cb=lambda x, name=f.name: bar.progress(x, text=f"🧠 Embedding {name} ... {int(x * 100)}%"))
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            st.error(f"❌ {f.name}: indexing failed ({exc}).")
            continue
        db.set_chunk_count(doc_id, n)
        bar.empty()
        st.success(f"✅ **{f.name}** uploaded — {len(pages)} pages, {n} chunks stored in the vector database.")


def delete_doc(doc):
    try:
        rag.delete_document(doc["id"])
    except Exception:  # noqa: BLE001
        pass
    Path(doc["file_path"]).unlink(missing_ok=True)
    db.delete_document(doc["id"], uid())
    if st.session_state.get("reader_doc_id") == doc["id"]:
        st.session_state.pop("reader_doc_id", None)
        st.session_state.pop("reader_doc", None)


def library_page():
    st.title("📁 My Library")
    st.caption("Upload lecture notes, books or any PDF. Each PDF is saved, its text is split into chunks "
               "and stored in the vector database for AI Search and the Chatbot.")

    with st.container(border=True):
        files = st.file_uploader("Choose PDF file(s)", type=["pdf"], accept_multiple_files=True)
        if st.button("⬆️ Upload & Process", type="primary", disabled=not files):
            process_uploads(files)

    docs = db.get_documents(uid())
    st.subheader(f"📚 Your PDFs ({len(docs)})")
    if not docs:
        st.info("No PDFs uploaded yet.")
    for d in docs:
        with st.container(border=True):
            a, b, c = st.columns([6, 1, 1], vertical_alignment="center")
            a.markdown(f"**📄 {d['file_name']}**")
            a.caption(f"{d['page_count']} pages · {d['chunk_count']} chunks · uploaded {d['uploaded_at']}")
            b.button("📖 Open", key=f"lib_open_{d['id']}", on_click=open_in_reader,
                     args=(d["id"], db.get_progress(uid(), d["id"])))
            with c.popover("🗑️"):
                st.write("Delete this PDF and all its data?")
                if st.button("Yes, delete", key=f"lib_del_{d['id']}", type="primary"):
                    delete_doc(d)
                    st.rerun()
            library_ocr(d)


@st.cache_data(show_spinner=False)
def cached_pages_without_text(path, mtime):
    return pdf_utils.pages_without_text(path)


def library_ocr(doc):
    """AI VISION OCR: read scanned pages (images) and add their text to the vector database."""
    path = Path(doc["file_path"])
    if not path.exists():
        return
    empty = cached_pages_without_text(str(path), path.stat().st_mtime)
    if not empty:
        return
    done = db.get_ocr_pages(doc["id"])
    todo = [p for p in empty if p not in done]
    with st.expander(f"👁️ Read scanned pages with AI Vision — {len(empty) - len(todo)} of {len(empty)} "
                     f"image-only pages done"):
        if not todo:
            st.success("All scanned pages are read ✅ — AI Search, Chatbot and Summary can use them.")
            return
        st.caption("The AI looks at each page image, writes out its text (OCR), and the text is stored in "
                   "the database and the vector database. Free AI plans have limits, so read a few pages at a time "
                   "— you can continue later.")
        count = st.number_input("Pages to read now", 1, len(todo), min(5, len(todo)), key=f"ocr_n_{doc['id']}")
        if st.button(f"👁️ Read next {int(count)} page(s)", key=f"ocr_go_{doc['id']}", type="primary"):
            run_ocr(doc, todo[:int(count)])
            st.rerun()


def run_ocr(doc, pages):
    llm = get_vision_llm()
    if not llm:
        return
    load_embedder()
    bar = st.progress(0.0, text="👁️ Reading pages with AI Vision...")
    ok = 0
    for i, p in enumerate(pages, 1):
        bar.progress((i - 1) / len(pages), text=f"👁️ Reading page {p} ({i}/{len(pages)}) with {llm.label}...")
        try:
            text = ai.ocr_image(llm, pdf_utils.page_to_png(doc["file_path"], p))
        except LLMError as exc:
            st.error(f"⚠️ Stopped at page {p}: {exc}")
            break
        db.save_ocr_text(doc["id"], p, text, llm.label)
        try:
            rag.index_document(doc["id"], uid(), doc["file_name"], [(p, text)])
        except Exception as exc:  # noqa: BLE001
            st.error(f"⚠️ Page {p} was read but could not be indexed: {exc}")
        ok += 1
    bar.empty()
    if ok:
        db.set_chunk_count(doc["id"], len(rag.get_collection().get(where={"doc_id": int(doc["id"])},
                                                                     include=[])["ids"]))
        st.toast(f"✅ {ok} page(s) read by AI Vision and added to the vector database")


# ===========================================================================
# 📖 PDF Reader (+ Finder, Bookmarks, Notes, Page tools)
# ===========================================================================
def reader_page():
    docs, by_id = doc_map()
    if need_documents(docs):
        return
    ids = [d["id"] for d in docs]

    if st.session_state.get("reader_doc_id") not in ids:
        st.session_state.reader_doc_id = ids[0]
        st.session_state.reader_page = db.get_progress(uid(), ids[0])
    if st.session_state.get("reader_doc") not in ids:
        st.session_state.reader_doc = st.session_state.reader_doc_id

    st.selectbox("📄 Document", ids, key="reader_doc", on_change=_reader_doc_changed,
                 format_func=lambda i: by_id[i]["file_name"])
    doc = by_id[st.session_state.reader_doc_id]
    if not doc_file_ok(doc):
        return
    path, n = doc["file_path"], max(1, doc["page_count"])

    page = min(max(1, int(st.session_state.get("reader_page", 1))), n)
    st.session_state.reader_page = page
    db.set_progress(uid(), doc["id"], page)

    left, right = st.columns([3, 2], gap="large")

    # ---------------- page viewer ----------------
    with left:
        c1, c2, c3, c4 = st.columns([1, 1.3, 1, 2], vertical_alignment="bottom")
        c1.button("◀ Prev", on_click=_change_page, args=(-1, n), disabled=page <= 1, width="stretch")
        st.session_state.page_jump = page
        c2.number_input(f"Page (1-{n})", min_value=1, max_value=n, step=1, key="page_jump", on_change=_jump_page)
        c3.button("Next ▶", on_click=_change_page, args=(1, n), disabled=page >= n, width="stretch")
        zoom = c4.select_slider("🔎 Zoom", options=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5], value=1.5, key="zoom")

        highlight = st.session_state.get("highlight_term", "")
        if highlight:
            h1, h2 = st.columns([4, 1], vertical_alignment="center")
            h1.caption(f"🟨 Highlighting: **{highlight}**")
            h2.button("✖ Clear", on_click=_clear_highlight, key="clear_hl")

        try:
            img = pdf_utils.render_page(path, page, zoom=zoom, highlight_terms=[highlight] if highlight else None)
            st.image(img, caption=f"{doc['file_name']} — Page {page} of {n}", width="stretch")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not show this page: {exc}")

    # ---------------- tools ----------------
    with right:
        t_find, t_vision, t_tools, t_book, t_note = st.tabs(
            ["🔍 Finder", "👁️ Ask AI (Vision)", "🤖 Page Tools", "🔖 Bookmarks", "🗒️ Notes"])
        with t_vision:
            reader_vision(doc, path, page)

        with t_find:
            reader_finder(doc, path)
        with t_book:
            reader_bookmarks(doc, page)
        with t_note:
            reader_notes(doc, page)
        with t_tools:
            reader_page_tools(doc, path, page)


def reader_finder(doc, path):
    st.caption("Find an exact word or phrase inside this PDF (like Ctrl+F).")
    with st.form("finder_form"):
        term = st.text_input("Word / phrase", placeholder="e.g. gradient descent")
        submitted = st.form_submit_button("🔍 Find", type="primary")
    if submitted and term.strip():
        results = pdf_utils.find_in_pdf(path, term)
        # also search text that AI Vision read from scanned pages
        found_pages = {r["page"] for r in results}
        pattern = re.compile(re.escape(term.strip()), re.IGNORECASE)
        for p, text in sorted(db.get_ocr_pages(doc["id"]).items()):
            hits = list(pattern.finditer(text))
            if hits and p not in found_pages:
                m = hits[0]
                snippet = text[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")
                results.append({"page": p, "count": len(hits), "snippets": [f"👁️ (scanned) ...{snippet}..."]})
        results.sort(key=lambda r: r["page"])
        st.session_state.finder = {"doc_id": doc["id"], "term": term.strip(), "results": results}
        st.session_state.highlight_term = term.strip()
        db.add_search(uid(), term.strip(), "keyword", sum(r["count"] for r in results))
        st.rerun()

    found = st.session_state.get("finder")
    if not found or found["doc_id"] != doc["id"]:
        return
    results = found["results"]
    if not results:
        st.warning(f"“{found['term']}” was not found in this PDF. Try **🔍 AI Search** to search by meaning.")
        return
    total = sum(r["count"] for r in results)
    st.success(f"Found **{total}** time(s) on **{len(results)}** page(s).")
    for r in results[:50]:
        with st.container(border=True):
            st.button(f"📄 Page {r['page']}  ·  {r['count']}×", key=f"find_{r['page']}",
                      on_click=open_in_reader, args=(doc["id"], r["page"], found["term"]))
            for s in r["snippets"]:
                st.caption(s)


def reader_vision(doc, path, page):
    """AI VISION on the current page: the AI looks at the page image (text, diagrams, charts, photos)."""
    st.caption("The AI **looks at this page as an image** — it can read scanned text, diagrams, charts, "
               "tables and pictures that normal text extraction cannot.")
    lang = answer_language()
    result_key = f"vis_{doc['id']}_{page}"

    q1, q2 = st.columns(2)
    action = None
    if q1.button("📄 Read this page (OCR)", width="stretch", key="vis_ocr"):
        action = ("ocr", None)
    if q2.button("📊 Explain diagrams / images", width="stretch", key="vis_diag"):
        action = ("ask", "Explain every diagram, chart, table or picture on this page in simple words. "
                         "If there are none, summarize the page.")
    with st.form("vision_form", clear_on_submit=False):
        question = st.text_input("Ask anything about this page", placeholder="e.g. What does the figure show?")
        if st.form_submit_button("👁️ Ask AI", type="primary") and question.strip():
            action = ("ask", question.strip())

    if action:
        llm = get_vision_llm()
        if llm:
            image = pdf_utils.page_to_png(path, page)
            if action[0] == "ocr":
                res = run_ai(f"👁️ {llm.label} is reading the page...", ai.ocr_image, llm, image)
                if res and not pdf_utils.get_page_text(path, page).strip():
                    # scanned page: save the text so Search / Chatbot / Summary can use it
                    db.save_ocr_text(doc["id"], page, res, llm.label)
                    try:
                        load_embedder()
                        rag.index_document(doc["id"], uid(), doc["file_name"], [(page, res)])
                        st.toast("Saved — this page is now searchable ✅")
                    except Exception:  # noqa: BLE001
                        pass
                task = "Read page (OCR)"
            else:
                res = run_ai(f"👁️ {llm.label} is looking at the page...", ai.vision_ask, llm, image, action[1], lang)
                task = action[1]
            if res:
                db.add_vision(uid(), f"{doc['file_name']} p.{page}", task[:300], res, llm.label)
                st.session_state[result_key] = {"task": task, "answer": res}

    data = st.session_state.get(result_key)
    if data:
        st.caption(f"**{data['task'][:120]}**")
        with st.container(border=True, height=420):
            st.markdown(data["answer"])
        a, b = st.columns(2)
        with a:
            listen_button(data["answer"], result_key)
        with b:
            download_button(data["answer"], f"page{page}_vision.txt", f"dl_{result_key}")


def reader_bookmarks(doc, page):
    marks = db.get_bookmarks(uid(), doc["id"])
    is_marked = any(b["page"] == page for b in marks)
    if is_marked:
        if st.button(f"❌ Remove bookmark (page {page})"):
            db.remove_bookmark(uid(), doc["id"], page)
            st.rerun()
    else:
        label = st.text_input("Label (optional)", key=f"bm_label_{doc['id']}_{page}", placeholder="e.g. Important formula")
        if st.button(f"🔖 Bookmark page {page}", type="primary"):
            db.add_bookmark(uid(), doc["id"], page, label)
            st.rerun()

    st.divider()
    if not marks:
        st.caption("No bookmarks yet.")
    for b in marks:
        st.button(f"🔖 Page {b['page']}" + (f" — {b['label']}" if b["label"] else ""),
                  key=f"bm_go_{b['id']}", on_click=_set_page, args=(b["page"],), width="stretch")


def reader_notes(doc, page):
    key = f"note_{doc['id']}_{page}"
    if key not in st.session_state:
        st.session_state[key] = db.get_note(uid(), doc["id"], page)
    st.text_area(f"My note for page {page}", key=key, height=160, placeholder="Write your own note here...")
    if st.button("💾 Save note", type="primary"):
        db.save_note(uid(), doc["id"], page, st.session_state[key])
        st.toast("Note saved ✅")

    notes = db.get_notes(uid(), doc["id"])
    if notes:
        with st.expander(f"📒 All notes in this PDF ({len(notes)})"):
            for nt in notes:
                st.button(f"Page {nt['page']}", key=f"note_go_{nt['id']}", on_click=_set_page, args=(nt["page"],))
                st.write(nt["content"])
            export = "\n\n".join(f"Page {nt['page']}:\n{nt['content']}" for nt in notes)
            download_button(export, f"notes_{Path(doc['file_name']).stem}.txt", "dl_notes", "⬇️ Download notes")


def reader_page_tools(doc, path, page):
    page_text = doc_page_text(doc, page)
    if not page_text:
        st.info("This page has no selectable text. Open the **👁️ Ask AI (Vision)** tab and press "
                "**Read this page (OCR)** first.")
        return

    tool = st.radio("Tool", ["📝 Summarize", "🌐 Translate", "💡 Explain simply", "🔊 Listen"],
                    horizontal=True, key="page_tool", label_visibility="collapsed")
    lang = answer_language()

    if tool == "🔊 Listen":
        st.caption("Listen to the text of this page.")
        listen_button(page_text, f"page_{doc['id']}_{page}", label="▶️ Read this page aloud")
        return

    direction = None
    if tool == "🌐 Translate":
        choice = st.radio("Direction", ["English → বাংলা", "বাংলা → English"], horizontal=True, key="page_tr_dir")
        direction = "en-bn" if choice.startswith("English") else "bn-en"

    result_key = f"pt_{doc['id']}_{page}_{tool}_{direction}_{lang}"
    if st.button(f"Run on page {page}", type="primary"):
        llm = get_llm()
        if llm:
            if tool == "📝 Summarize":
                res = run_ai("Summarizing page...", ai.summarize, llm, page_text, "Bullet points", lang)
            elif tool == "🌐 Translate":
                res = run_ai("Translating page...", ai.translate, llm, page_text, direction)
                if res:
                    db.add_translation(uid(), direction, page_text[:2000], res)
            else:
                res = run_ai("Explaining...", ai.explain_simply, llm, page_text, lang)
            if res:
                st.session_state[result_key] = res

    if result_key in st.session_state:
        res = st.session_state[result_key]
        with st.container(border=True, height=420):
            st.markdown(res)
        a, b = st.columns(2)
        with a:
            listen_button(res, result_key)
        with b:
            download_button(res, f"page{page}_{tool.split()[-1].lower()}.txt", f"dl_{result_key}")


# ===========================================================================
# 🔍 AI Search  (semantic search with the vector database)
# ===========================================================================
def ai_search_page():
    st.title("🔍 AI Search")
    st.caption("Search by **meaning**, not exact words. Your question is turned into a vector and compared "
               "with every chunk of your PDFs in the vector database (cosine similarity).")
    docs, by_id = doc_map()
    if need_documents(docs):
        return

    with st.form("ai_search_form"):
        query = st.text_input("What are you looking for? (English or বাংলা)",
                              placeholder="e.g. how does a neural network learn?")
        c1, c2 = st.columns([3, 1])
        scope = c1.multiselect("Search in", [d["id"] for d in docs], format_func=lambda i: by_id[i]["file_name"],
                               placeholder="All my PDFs")
        k = c2.slider("Results", 3, 10, config.TOP_K)
        o1, o2 = st.columns(2)
        translate_bn = o1.checkbox("Translate Bangla query to English first", value=True,
                                   help="Gives better results when your PDF is in English (uses the AI provider).")
        compare = o2.checkbox("Compare with keyword search (Finder)", value=False)
        submitted = st.form_submit_button("🔍 Search", type="primary")

    if submitted and query.strip():
        load_embedder()
        search_q = query.strip()
        if translate_bn and ai.has_bangla(search_q):
            search_q = ai.query_for_search(get_llm(show_warning=False), search_q)
        try:
            results = rag.search(search_q, uid(), scope or None, k)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Search failed: {exc}")
            return
        keyword = []
        if compare:
            for d in docs:
                if (not scope or d["id"] in scope) and Path(d["file_path"]).exists():
                    for r in pdf_utils.find_in_pdf(d["file_path"], query.strip(), max_snippets=1):
                        keyword.append({**r, "doc_id": d["id"], "file_name": d["file_name"]})
        db.add_search(uid(), query.strip(), "ai", len(results))
        st.session_state.ai_search = {"query": query.strip(), "search_q": search_q,
                                      "results": results, "keyword": keyword, "compare": compare}

    data = st.session_state.get("ai_search")
    if not data:
        return
    if data["search_q"] != data["query"]:
        st.caption(f"🌐 Searched as: *{data['search_q']}*")

    if data["compare"]:
        col_ai, col_kw = st.columns(2)
    else:
        col_ai, col_kw = st.container(), None

    with col_ai:
        st.subheader(f"🧠 AI (semantic) results — {len(data['results'])}")
        if not data["results"]:
            st.info("No results.")
        for i, r in enumerate(data["results"], 1):
            with st.container(border=True):
                st.markdown(f"**#{i} · {r['file_name']} · Page {r['page']}**")
                st.progress(min(1.0, r["score"]), text=f"Similarity: {r['score'] * 100:.0f}%")
                text = r["text"]
                st.write(text[:700] + ("..." if len(text) > 700 else ""))
                st.button("📖 Open this page", key=f"ais_open_{i}", on_click=open_in_reader,
                          args=(r["doc_id"], r["page"]))

    if col_kw is not None:
        with col_kw:
            hits = data["keyword"]
            st.subheader(f"🔤 Keyword results — {sum(h['count'] for h in hits)} match(es)")
            if not hits:
                st.info("Exact words not found. This is why AI Search is useful — it understands meaning!")
            for j, h in enumerate(hits[:20]):
                with st.container(border=True):
                    st.markdown(f"**{h['file_name']} · Page {h['page']}** ({h['count']}×)")
                    for s in h["snippets"]:
                        st.caption(s)
                    st.button("📖 Open", key=f"kw_open_{j}", on_click=open_in_reader,
                              args=(h["doc_id"], h["page"], data["query"]))


# ===========================================================================
# 💬 AI Chatbot  (RAG)
# ===========================================================================
SUGGESTIONS = [
    "What is this document about?",
    "List the most important topics.",
    "Explain the main concept in simple words.",
    "এই PDF-এর মূল বিষয়গুলো কী?",
]


def chatbot_page():
    st.title("💬 AI Chatbot")
    st.caption("Ask questions about your PDFs. The answer is generated **only from your PDFs** (RAG) "
               "and shows the source pages. Click a source to open that page.")
    docs, by_id = doc_map()
    if need_documents(docs):
        return

    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    scope = c1.multiselect("Chat with", [d["id"] for d in docs], format_func=lambda i: by_id[i]["file_name"],
                           placeholder="All my PDFs", key="chat_scope")
    if c2.button("🧹 New chat", width="stretch"):
        st.session_state.chat_messages = []
        st.rerun()

    messages = st.session_state.setdefault("chat_messages", [])

    if not messages:
        st.markdown("**Try asking:**")
        cols = st.columns(len(SUGGESTIONS))
        for col, s in zip(cols, SUGGESTIONS):
            col.button(s, on_click=_ask, args=(s,), width="stretch")

    for idx, m in enumerate(messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_assistant_extras(idx, m)

    question = st.chat_input("Ask anything about your PDFs... (English or বাংলা)")
    question = question or st.session_state.pop("pending_q", None)
    if not question:
        return

    llm = get_llm()
    if not llm:
        return
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("🔎 Searching your PDFs and thinking..."):
                load_embedder()
                search_q = ai.query_for_search(llm, question)
                contexts = rag.search(search_q, uid(), scope or None, config.TOP_K)
                history = [{"role": x["role"], "content": x["content"]} for x in messages][-6:]
                answer = ai.answer_question(llm, question, contexts, history, answer_language())
        except LLMError as exc:
            st.error(f"⚠️ {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"⚠️ Something went wrong: {exc}")
            return

    sources, seen = [], set()
    for c in contexts:
        if (c["doc_id"], c["page"]) not in seen:
            seen.add((c["doc_id"], c["page"]))
            sources.append({"doc_id": c["doc_id"], "file_name": c["file_name"], "page": c["page"]})
    chat_id = db.add_chat(uid(), question, answer, sources, llm.label)

    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer, "sources": sources,
                     "contexts": contexts, "chat_id": chat_id, "feedback": None})
    st.rerun()


def render_assistant_extras(idx, m):
    sources = m.get("sources") or []
    if sources:
        st.caption("📚 Sources:")
        cols = st.columns(min(4, len(sources)))
        for j, s in enumerate(sources):
            name = s["file_name"] if len(s["file_name"]) <= 22 else s["file_name"][:20] + "…"
            cols[j % len(cols)].button(f"📄 {name} · p.{s['page']}", key=f"src_{idx}_{j}",
                                       on_click=open_in_reader, args=(s["doc_id"], s["page"]))

    f1, f2, f3, _ = st.columns([1, 1, 2, 4])
    fb = m.get("feedback")
    f1.button("👍" + (" ✓" if fb == 1 else ""), key=f"up_{idx}", on_click=_set_feedback, args=(idx, m["chat_id"], 1))
    f2.button("👎" + (" ✓" if fb == -1 else ""), key=f"down_{idx}", on_click=_set_feedback, args=(idx, m["chat_id"], -1))
    with f3:
        listen_button(m["content"], f"chat_{idx}")

    if m.get("contexts"):
        with st.expander("🔎 Retrieved chunks (RAG context sent to the AI)"):
            for c in m["contexts"]:
                st.markdown(f"**{c['file_name']} · Page {c['page']} · similarity {c['score'] * 100:.0f}%**")
                st.caption(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))


# ===========================================================================
# 📝 Summary
# ===========================================================================
def summary_page():
    st.title("📝 Summary")
    st.caption("Get a summary of a whole PDF or only some pages. Summaries are saved in the database.")
    docs, by_id = doc_map()
    if need_documents(docs):
        return

    doc_id = st.selectbox("📄 Document", [d["id"] for d in docs], format_func=lambda i: by_id[i]["file_name"], key="sum_doc")
    doc = by_id[doc_id]
    if not doc_file_ok(doc):
        return
    n = max(1, doc["page_count"])

    c1, c2, c3 = st.columns(3)
    scope_type = c1.radio("Summarize", ["Whole document", "Page range"], key="sum_scope")
    style = c2.selectbox("Style", ["Short", "Detailed", "Bullet points"], key="sum_style")
    language = c3.selectbox("Language", LANGUAGES, index=LANGUAGES.index(answer_language()), key="sum_lang")

    if scope_type == "Page range":
        r1, r2 = st.columns(2)
        start = r1.number_input("From page", 1, n, 1, key="sum_from")
        end = r2.number_input("To page", 1, n, min(n, 5), key="sum_to")
        if end < start:
            st.warning("'To page' must be after 'From page'.")
            return
        scope = f"{start}-{end}"
    else:
        start, end, scope = 1, n, "all"

    cached = db.get_summary(uid(), doc_id, scope, style, language)
    if st.button("🔄 Regenerate summary" if cached else "✨ Generate summary", type="primary"):
        llm = get_llm()
        if llm:
            text = doc_pages_text(doc, start, end)
            result = run_ai("Reading the PDF and writing the summary... (big PDFs take longer)",
                            ai.summarize, llm, text, style, language)
            if result:
                db.save_summary(uid(), doc_id, scope, style, language, result)
                cached = db.get_summary(uid(), doc_id, scope, style, language)

    if not cached:
        return
    st.caption(f"Saved {cached['created_at']} · pages: {scope} · {style} · {language}")
    with st.container(border=True):
        st.markdown(cached["content"])

    a, b, c, d = st.columns(4)
    with a:
        listen_button(cached["content"], f"sum_{cached['id']}")
    with b:
        download_button(cached["content"], f"summary_{Path(doc['file_name']).stem}.txt", f"dl_sum_{cached['id']}")
    tr_key = f"sum_tr_{cached['id']}"
    topics_key = f"sum_topics_{cached['id']}"
    other = "বাংলা" if language == "English" else "English"
    if c.button(f"🌐 Translate to {other}"):
        llm = get_llm()
        if llm:
            res = run_ai("Translating...", ai.translate, llm, cached["content"],
                         "en-bn" if language == "English" else "bn-en")
            if res:
                st.session_state[tr_key] = res
    if d.button("🏷️ Key topics"):
        llm = get_llm()
        if llm:
            text = doc_pages_text(doc, start, end)
            res = run_ai("Finding key topics...", ai.key_topics, llm, text, language)
            if res:
                st.session_state[topics_key] = res

    if tr_key in st.session_state:
        st.subheader(f"🌐 Translation ({other})")
        with st.container(border=True):
            st.markdown(st.session_state[tr_key])
    if topics_key in st.session_state:
        st.subheader("🏷️ Key topics")
        with st.container(border=True):
            st.markdown(st.session_state[topics_key])


# ===========================================================================
# 🌐 Translator
# ===========================================================================
def _load_page_into_translator(doc):
    st.session_state.tr_input = doc_page_text(doc, int(st.session_state.tr_page))


def translator_page():
    st.title("🌐 Translator")
    st.caption("Translate English ⇄ বাংলা. Type/paste text or load the text of a PDF page.")

    choice = st.radio("Direction", ["English → বাংলা", "বাংলা → English"], horizontal=True, key="tr_dir")
    direction = "en-bn" if choice.startswith("English") else "bn-en"

    docs, by_id = doc_map()
    if docs:
        with st.expander("📄 Load text from a PDF page"):
            c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
            doc_id = c1.selectbox("Document", [d["id"] for d in docs], format_func=lambda i: by_id[i]["file_name"], key="tr_doc")
            doc = by_id[doc_id]
            c2.number_input("Page", 1, max(1, doc["page_count"]), 1, key="tr_page")
            c3.button("Load", on_click=_load_page_into_translator, args=(doc,), width="stretch")

    text = st.text_area("Text to translate", key="tr_input", height=200,
                        placeholder="Write or paste text here...")
    if st.button("🌐 Translate", type="primary", disabled=not text.strip()):
        llm = get_llm()
        if llm:
            res = run_ai("Translating...", ai.translate, llm, text.strip(), direction)
            if res:
                db.add_translation(uid(), direction, text.strip()[:2000], res)
                st.session_state.tr_result = {"source": text.strip(), "result": res, "direction": direction}

    data = st.session_state.get("tr_result")
    if data:
        left, right = st.columns(2)
        src_label, tgt_label = ("English", "বাংলা") if data["direction"] == "en-bn" else ("বাংলা", "English")
        with left:
            st.markdown(f"**Original ({src_label})**")
            with st.container(border=True, height=350):
                st.write(data["source"])
        with right:
            st.markdown(f"**Translation ({tgt_label})**")
            with st.container(border=True, height=350):
                st.write(data["result"])
        a, b, _ = st.columns([1, 1, 2])
        with a:
            listen_button(data["result"], "tr_result", lang="bn" if data["direction"] == "en-bn" else "en")
        with b:
            download_button(data["result"], "translation.txt", "dl_tr")

    recent = db.get_translations(uid(), 10)
    if recent:
        with st.expander(f"🕘 Recent translations ({len(recent)})"):
            for t in recent:
                st.caption(f"{t['created_at']} · {t['direction']}")
                st.markdown(f"> {t['source_text'][:200]}{'...' if len(t['source_text']) > 200 else ''}")
                st.markdown(t["translated"][:300] + ("..." if len(t["translated"]) > 300 else ""))
                st.divider()


# ===========================================================================
# 🧠 Study Tools: Explain simply, Word meaning, Quiz
# ===========================================================================
def study_tools_page():
    st.title("🧠 Study Tools")
    t1, t2, t3, t4 = st.tabs(["💡 Explain Simply", "📖 Word Meaning", "🧪 Quiz Generator", "🏆 Quiz Scores"])

    with t1:
        st.caption("Paste a difficult paragraph and get an easy explanation.")
        text = st.text_area("Difficult text", height=160, key="explain_input")
        if st.button("💡 Explain", type="primary", disabled=not text.strip()):
            llm = get_llm()
            if llm:
                res = run_ai("Explaining...", ai.explain_simply, llm, text, answer_language())
                if res:
                    st.session_state.explain_result = res
        if st.session_state.get("explain_result"):
            with st.container(border=True):
                st.markdown(st.session_state.explain_result)
            listen_button(st.session_state.explain_result, "explain")

    with t2:
        st.caption("Meaning of a word in English and বাংলা, with an example.")
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        word = c1.text_input("Word or term", key="word_input", placeholder="e.g. photosynthesis")
        if c2.button("📖 Find meaning", type="primary", disabled=not word.strip(), width="stretch"):
            llm = get_llm()
            if llm:
                res = run_ai("Looking up...", ai.word_meaning, llm, word.strip())
                if res:
                    st.session_state.word_result = res
        if st.session_state.get("word_result"):
            with st.container(border=True):
                st.markdown(st.session_state.word_result)

    with t3:
        quiz_tab()

    with t4:
        results = db.get_quiz_results(uid())
        if not results:
            st.info("No quizzes taken yet.")
        else:
            st.dataframe(
                [{"Date": r["created_at"], "PDF": r["doc_name"], "Score": f"{r['score']}/{r['total']}",
                  "Percent": f"{round(100 * r['score'] / max(1, r['total']))}%"} for r in results],
                width="stretch", hide_index=True)


def quiz_tab():
    docs, by_id = doc_map()
    if need_documents(docs):
        return
    st.caption("Create multiple-choice questions from your PDF and test yourself. Scores are saved.")
    c1, c2, c3 = st.columns([3, 1, 1])
    doc_id = c1.selectbox("📄 Document", [d["id"] for d in docs], format_func=lambda i: by_id[i]["file_name"], key="quiz_doc")
    doc = by_id[doc_id]
    n = max(1, doc["page_count"])
    count = c2.number_input("Questions", 3, 15, 5, key="quiz_count")
    lang = c3.selectbox("Language", LANGUAGES, index=LANGUAGES.index(answer_language()), key="quiz_lang")
    r1, r2 = st.columns(2)
    start = r1.number_input("From page", 1, n, 1, key="quiz_from")
    end = r2.number_input("To page", 1, n, min(n, 10), key="quiz_to")

    if st.button("🧪 Generate quiz", type="primary", disabled=end < start):
        llm = get_llm()
        if llm and doc_file_ok(doc):
            text = doc_pages_text(doc, start, end)
            if not text.strip():
                st.warning("These pages have no text.")
            else:
                questions = run_ai("Creating questions...", ai.generate_quiz, llm, text, int(count), lang)
                if questions:
                    st.session_state.quiz_counter = st.session_state.get("quiz_counter", 0) + 1
                    st.session_state.quiz = {"id": st.session_state.quiz_counter, "doc_id": doc_id,
                                             "doc_name": doc["file_name"], "questions": questions, "result": None}

    quiz = st.session_state.get("quiz")
    if not quiz:
        return
    st.divider()
    st.subheader(f"🧪 Quiz: {quiz['doc_name']}")
    with st.form(f"quiz_form_{quiz['id']}"):
        for i, q in enumerate(quiz["questions"]):
            st.radio(f"**Q{i + 1}. {q['question']}**", list(range(len(q["options"]))),
                     format_func=lambda j, q=q: q["options"][j], index=None, key=f"quiz_{quiz['id']}_{i}")
        submitted = st.form_submit_button("✅ Submit answers", type="primary", disabled=quiz["result"] is not None)

    if submitted and quiz["result"] is None:
        answers = [st.session_state.get(f"quiz_{quiz['id']}_{i}") for i in range(len(quiz["questions"]))]
        score = sum(1 for a, q in zip(answers, quiz["questions"]) if a == q["answer"])
        quiz["result"] = {"answers": answers, "score": score}
        db.add_quiz_result(uid(), quiz["doc_id"], quiz["doc_name"], score, len(quiz["questions"]))

    if quiz["result"]:
        score, total = quiz["result"]["score"], len(quiz["questions"])
        pct = round(100 * score / total)
        (st.balloons() if pct >= 80 else None)
        st.success(f"🏆 Your score: **{score} / {total}** ({pct}%)")
        for i, (q, a) in enumerate(zip(quiz["questions"], quiz["result"]["answers"])):
            correct = q["options"][q["answer"]]
            if a == q["answer"]:
                st.markdown(f"✅ **Q{i + 1}.** Correct — {correct}")
            else:
                yours = q["options"][a] if a is not None else "no answer"
                st.markdown(f"❌ **Q{i + 1}.** Your answer: {yours} · Correct: **{correct}**")
            if q["explanation"]:
                st.caption(f"💡 {q['explanation']}")


# ===========================================================================
# 📷 Image AI (AI Vision on photos / screenshots)
# ===========================================================================
def image_ai_page():
    st.title("📷 Image AI")
    st.caption("**AI Vision (multimodal AI):** upload a photo of your notes, a book page, the whiteboard, "
               "a diagram or a question — the AI looks at the image and reads, explains, translates or solves it.")

    src = st.radio("Image source", ["📁 Upload image", "📸 Take photo (camera)"], horizontal=True, key="img_src")
    if src.startswith("📁"):
        file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp", "bmp", "gif"], key="img_file")
    else:
        file = st.camera_input("Take a photo", key="img_cam")
    if not file:
        st.info("Add an image to start.")
        return

    try:
        image = pdf_utils.image_to_png(file.getvalue())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not open this image: {exc}")
        return

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.image(image, caption=file.name if hasattr(file, "name") else "Photo", width="stretch")
    with right:
        task = st.radio("What should the AI do?", list(ai.IMAGE_TASKS.keys()), key="img_task")
        extra = st.text_input("Extra instruction / your question (optional)",
                              placeholder="e.g. only the second question / explain for a beginner", key="img_extra")
        if st.button("👁️ Run AI Vision", type="primary"):
            llm = get_vision_llm()
            if llm:
                res = run_ai(f"👁️ {llm.label} is looking at the image...", ai.image_task, llm, image, task,
                             answer_language(), extra)
                if res:
                    db.add_vision(uid(), "image", task, res, llm.label)
                    st.session_state.img_result = {"img": _hash(str(len(image)) + task), "task": task,
                                                   "answer": res, "provider": llm.label}

    data = st.session_state.get("img_result")
    if data:
        st.divider()
        st.subheader(data["task"])
        st.caption(f"AI: {data['provider']}")
        with st.container(border=True):
            st.markdown(data["answer"])
        a, b, _ = st.columns([1, 1, 2])
        with a:
            listen_button(data["answer"], "img_result")
        with b:
            download_button(data["answer"], "image_ai_result.txt", "dl_img")


# ===========================================================================
# 🕘 History
# ===========================================================================
def history_page():
    st.title("🕘 History")
    t1, t2, t3, t5, t4 = st.tabs(["💬 Chat history", "🔍 Search history", "🌐 Translations",
                                  "👁️ Vision history", "📁 Upload history"])

    with t5:
        items = db.get_vision_history(uid())
        if not items:
            st.info("No AI Vision tasks yet. Try **📷 Image AI** or the **👁️ Ask AI** tab in the PDF Reader.")
        for v in items:
            with st.expander(f"{v['created_at']} · {v['source']} · {v['task'][:60]}"):
                st.markdown(v["answer"])
                st.caption(f"AI: {v['provider']}")

    with t1:
        chats = db.get_chats(uid())
        if not chats:
            st.info("No chat history yet.")
        else:
            a, b, _ = st.columns([1, 1, 3])
            export = "\n\n".join(
                f"[{c['created_at']}] ({c['provider']})\nQ: {c['question']}\nA: {c['answer']}\nSources: "
                + ", ".join(f"{s['file_name']} p.{s['page']}" for s in c["sources"]) for c in reversed(chats))
            with a:
                download_button(export, "chat_history.txt", "dl_chats", "⬇️ Export all")
            with b.popover("🗑️ Clear all"):
                if st.button("Yes, delete all chats", type="primary"):
                    db.clear_chats(uid())
                    st.session_state.chat_messages = []
                    st.rerun()
            for c in chats:
                fb = {1: " 👍", -1: " 👎"}.get(c["feedback"], "")
                with st.expander(f"{c['created_at']} — {c['question'][:80]}{fb}"):
                    st.markdown(c["answer"])
                    if c["sources"]:
                        st.caption("Sources: " + ", ".join(f"{s['file_name']} p.{s['page']}" for s in c["sources"]))
                    st.caption(f"AI: {c['provider']}")
                    if st.button("🗑️ Delete", key=f"del_chat_{c['id']}"):
                        db.delete_chat(c["id"], uid())
                        st.rerun()

    with t2:
        searches = db.get_searches(uid())
        if not searches:
            st.info("No searches yet.")
        else:
            st.dataframe([{"Date": s["created_at"], "Query": s["query"],
                           "Type": "🧠 AI" if s["search_type"] == "ai" else "🔤 Keyword",
                           "Results": s["results"]} for s in searches],
                         width="stretch", hide_index=True)

    with t3:
        trs = db.get_translations(uid(), 100)
        if not trs:
            st.info("No translations yet.")
        for t in trs:
            with st.expander(f"{t['created_at']} · {t['direction']} · {t['source_text'][:60]}"):
                st.markdown("**Original:**")
                st.write(t["source_text"])
                st.markdown("**Translation:**")
                st.write(t["translated"])

    with t4:
        docs = db.get_documents(uid())
        if not docs:
            st.info("No uploads yet.")
        else:
            st.dataframe([{"Uploaded": d["uploaded_at"], "File": d["file_name"], "Pages": d["page_count"],
                           "Chunks (vectors)": d["chunk_count"]} for d in docs],
                         width="stretch", hide_index=True)


# ===========================================================================
# Main
# ===========================================================================
PAGE_FUNCS = {
    NAV_DASHBOARD: dashboard_page,
    NAV_LIBRARY: library_page,
    NAV_READER: reader_page,
    NAV_SEARCH: ai_search_page,
    NAV_CHAT: chatbot_page,
    NAV_SUMMARY: summary_page,
    NAV_TRANSLATE: translator_page,
    NAV_TOOLS: study_tools_page,
    NAV_IMAGE: image_ai_page,
    NAV_HISTORY: history_page,
}


def main():
    db.init_db()
    if "user" not in st.session_state:
        auth_page()
        return
    sidebar()
    PAGE_FUNCS.get(st.session_state.get("nav"), dashboard_page)()


main()
