# 📚 StudyMate AI — বাংলা গাইড (Setup ও Demo)

AI Engineering Final Project. User নিজের PDF upload করে পড়তে পারে, AI দিয়ে খুঁজতে পারে, প্রশ্ন করতে পারে, summary আর translation পেতে পারে।

| Requirement | কোথায় আছে |
|---|---|
| **AI Search Option** | 🔍 AI Search page → `core/rag.py` (`search`) |
| **AI Chatbot** | 💬 AI Chatbot page → `core/ai_features.py` (`answer_question`) + `core/llm.py` |
| **RAG / Vector Database** | ChromaDB + sentence-transformers → `core/rag.py` |
| **Database Connection** | SQLite → `core/database.py` (`data/studymate.db`) |

AI provider: **Google Gemini**, **Groq**, **Ollama** (Cloud বা নিজের Mac-এ) আর **Anthropic Claude** — app-এর sidebar থেকে যেকোনো সময় বদলানো যায়।

---

## ✨ Features

- 🔐 **Login / Register** — password `bcrypt` দিয়ে encrypt করা
- 📁 **My Library** — একসাথে কয়েকটা PDF upload, upload history, delete
- 📖 **PDF Reader** — page by page পড়া, zoom, last read page মনে রাখে
  - 🔍 **Finder** — PDF-এর ভেতরে হুবহু শব্দ খোঁজা, হলুদ highlight
  - 🔖 **Bookmarks** আর 🗒️ **Page notes**
  - 🤖 **Page tools** — এই page-এর summary, translation, সহজ ব্যাখ্যা, 🔊 শোনা
- 🔍 **AI Search** — অর্থ দিয়ে খোঁজা (semantic search), similarity % সহ, keyword search-এর সাথে তুলনা
- 💬 **AI Chatbot (RAG)** — PDF থেকে উত্তর, source page-এ click করলে সেই page খোলে, 👍👎 feedback, retrieved chunks দেখা যায়
- 📝 **Summary** — পুরো PDF বা page range, Short / Detailed / Bullet, বাংলা বা English, key topics
- 🌐 **Translator** — English ⇄ বাংলা, PDF page load করে translate
- 🧠 **Study Tools** — Explain simply, Word meaning (বাংলা + English), **Quiz generator** (MCQ + score)
- 👁️ **AI Vision (Multimodal AI)**
  - **Scanned PDF support** — ছবি/scan করা page AI দেখে লেখা পড়ে (OCR), তারপর AI Search, Chatbot, Summary-তে ব্যবহার হয়
  - **Ask AI about this page** — PDF Reader-এ খোলা page-এর diagram, chart, table নিয়ে প্রশ্ন
  - **📷 Image AI** — notes/board/বইয়ের ছবি upload বা camera দিয়ে তুলে: OCR, ব্যাখ্যা, অনুবাদ, প্রশ্নের সমাধান, study notes
- 🕘 **History** — chat, search, translation, upload history + export
- 🏠 **Dashboard** — statistics, continue reading

---

## 🍎 MacBook + PyCharm-এ চালানোর নিয়ম

### ধাপ ১: Python install
Python **3.11 বা 3.12** ব্যবহার করো (3.13-এ কিছু package সমস্যা করতে পারে)।
Terminal-এ check করো:
```bash
python3 --version
```
না থাকলে https://www.python.org/downloads/macos/ থেকে Python 3.12 install করো।
Install-এর পর **Applications → Python 3.12 → "Install Certificates.command"** double-click করো (না করলে model download-এ SSL error আসতে পারে)।

### ধাপ ২: PyCharm-এ project খোলা
1. ZIP file unzip করো।
2. PyCharm → **File → Open** → `StudyMate-AI` folder select করো।
3. PyCharm virtual environment বানাতে বললে **OK** দাও।
   না বললে: **PyCharm → Settings → Project: StudyMate-AI → Python Interpreter → Add Interpreter → Add Local Interpreter → Virtualenv → Base: Python 3.12 → OK**

### ধাপ ৩: Package install
PyCharm-এর নিচে **Terminal** tab খোলো (প্রথমে `(.venv)` লেখা থাকা উচিত), তারপর:
```bash
pip install -r requirements.txt
```
প্রথমবার ৫-১০ মিনিট লাগতে পারে (PyTorch বড় package)।

### ধাপ ৪: API key সেট করা
অন্তত **একটা** key লাগবে। একাধিক থাকলে একটার limit শেষ হলে অন্যটায় যাওয়া যায়।

| Provider | Key কোথায় পাবে | খরচ | Free limit (মোটামুটি) |
|---|---|---|---|
| **Groq** ⭐ | https://console.groq.com/keys | Free | ৩০/মিনিট, ~১,০০০/দিন |
| **Gemini** | https://aistudio.google.com/apikey | Free | ৫-১৫/মিনিট, model অনুযায়ী |
| **Ollama Cloud** | https://ollama.com/settings/keys | Free plan | সীমা প্রকাশিত নয় |
| **Claude** | https://console.anthropic.com/settings/keys | Paid | — |

1. Project-এর `.env.example` (Finder-এ লুকানো থাকলে `env_template.txt`) copy করে নাম দাও **`.env`**, তারপর key paste করো:
   ```
   GROQ_API_KEY=gsk_...
   GEMINI_API_KEY=AIza...
   OLLAMA_API_KEY=...
   DEFAULT_PROVIDER=Groq
   ```
   > `.env` না বানালেও চলবে — app চালিয়ে sidebar-এর **⚙️ AI Settings**-এ key paste করা যায়।

   Terminal দিয়ে copy করতে: `cp .env.example .env`

### ধাপ ৫: Run ▶️
`run.py` file খোলো → উপরের **সবুজ ▶ Run** button চাপো।
Browser-এ নিজে থেকে খুলবে: **http://localhost:8501**

(বিকল্প: Terminal-এ `streamlit run app.py`)

### ধাপ ৬: ব্যবহার
1. **Register** করে account বানাও → **Login**
2. **📁 My Library** → `sample_pdfs/AI_Lecture_Notes.pdf` upload করো (demo-র জন্য দেওয়া আছে)
   - প্রথম upload-এ AI search model (~470 MB) একবার download হবে, কয়েক মিনিট লাগতে পারে
3. বাকি সব page ব্যবহার করো 🎉

বন্ধ করতে: PyCharm-এ লাল **■ Stop** button।

---

## 🎤 Teacher-কে Demo দেখানোর ক্রম (৫-৭ মিনিট)

1. **Register/Login** → "user data SQLite database-এ save হয়, password encrypted"
2. **Upload** sample PDF → "text বের করে chunk করছে, embedding বানিয়ে ChromaDB-তে রাখছে"
3. **PDF Reader** → page পড়া, **Finder**-এ `gradient` খোঁজা → হলুদ highlight
4. **AI Search** → `how does a network learn?` লিখে **Compare with keyword search** tick দাও
   → keyword search কিছু পায় না, AI Search ঠিকই Lecture 2 খুঁজে দেয়। **এটাই semantic search!**
5. **AI Chatbot** → `What is RAG and why is it useful?` → উত্তরে source page → click করলে PDF-এর সেই page খোলে
   → **"Retrieved chunks"** expander খুলে দেখাও AI-কে কোন অংশ পাঠানো হয়েছে (RAG প্রমাণ)
   → বাংলায় প্রশ্ন করো: `নিউরাল নেটওয়ার্ক কীভাবে শেখে?`
6. Sidebar থেকে **Groq → Gemini → Ollama** বদলে একই প্রশ্ন করো (একাধিক AI provider support দেখাও)
7. **Summary**, **Translator**, **Quiz** দেখাও
8. **AI Vision** দেখাও:
   - **My Library**-তে `sample_pdfs/Scanned_RAG_Notes.pdf` upload করো → "এটা scanned, কোনো text নেই" →
     **👁️ Read scanned pages with AI Vision** → এখন AI Search-এ `vector database` খুঁজলে scanned page-ও পাওয়া যায়
   - PDF Reader-এ page 2 (diagram) খুলে **👁️ Ask AI → Explain diagrams / images**
   - **📷 Image AI**-তে খাতার একটা ছবি তুলে **Solve / answer the question** চাপো
9. **History** আর **Dashboard** → "সব কিছু database-এ save হচ্ছে"

---

## 🗂️ Project Structure

```
StudyMate-AI/
├── run.py               ← PyCharm-এ এটা Run করো
├── app.py               ← Streamlit UI (সব page)
├── config.py            ← settings (model, path, chunk size)
├── requirements.txt
├── .env.example         ← API key template
├── core/
│   ├── database.py      ← DATABASE CONNECTION (SQLite)
│   ├── rag.py           ← RAG / VECTOR DB + AI SEARCH (ChromaDB)
│   ├── llm.py           ← Gemini + Groq + Ollama + Claude
│   ├── ai_features.py   ← CHATBOT, summary, translate, quiz ...
│   ├── pdf_utils.py     ← PDF read, render, highlight, Finder
│   └── tts.py           ← Text to speech (বাংলা + English)
├── sample_pdfs/
│   ├── AI_Lecture_Notes.pdf     ← সাধারণ text PDF
│   └── Scanned_RAG_Notes.pdf    ← scanned (শুধু ছবি) — AI Vision demo-র জন্য
└── data/                ← app চালালে নিজে তৈরি হয়
    ├── studymate.db     ← SQLite database
    ├── chroma_db/       ← vector database
    └── uploads/         ← uploaded PDFs
```

## 🔄 How RAG works here

```
PDF ──PyMuPDF──► text ──chunk (900 chars)──► embedding (multilingual-e5-small)
                                                     │
                                                     ▼
                                              ChromaDB (vectors)
                                                     │  cosine similarity, top-5
Question ──► embedding ──────────────────────────────┘
                                                     │
                     top chunks + question ──► Gemini/Groq/Ollama/Claude ──► answer + pages
                                                                        │
                                                                        ▼
                                                               SQLite (chat_history)
```

## 👁️ How AI Vision works here

```
Scanned page / photo ──PyMuPDF──► PNG image ──► Vision model (Gemini / Claude)
                                                   │
                     ┌─────────────────────────────┼────────────────────────┐
                 OCR text                    Diagram explanation      Answer / translation
                     │
     saved in SQLite (ocr_pages) + chunked & embedded into ChromaDB → searchable by AI Search / Chatbot
```

Sidebar → ⚙️ AI Settings → **👁️ Vision AI**: ছবির কাজের জন্য আলাদা provider বাছা যায়। Default **Gemini** (free),
কারণ Groq/Ollama-র বেশিরভাগ model শুধু text পড়তে পারে। তাই chat-এর জন্য Groq আর vision-এর জন্য Gemini, একসাথে চলে।

## 🗄️ Database tables (SQLite)
`users`, `documents`, `reading_progress`, `bookmarks`, `notes`, `chat_history`, `search_history`, `summaries`, `translations`, `quiz_results`, `ocr_pages`, `vision_history`

PyCharm-এ দেখতে: ডানপাশে **Database** tab → **+ → Data Source → SQLite** → `data/studymate.db` select করো।
(Community edition-এ না থাকলে free app **DB Browser for SQLite** ব্যবহার করো।)

---

## 🛠️ সমস্যা ও সমাধান

| সমস্যা | সমাধান |
|---|---|
| `ModuleNotFoundError` | Terminal-এ `(.venv)` আছে কিনা দেখো, তারপর `pip install -r requirements.txt` |
| SSL / certificate error | Applications → Python 3.x → **Install Certificates.command** চালাও |
| "API key is invalid" | Key ঠিকভাবে copy হয়েছে কিনা দেখো, `.env`-এ space বা quote ছাড়া লেখো |
| "model not found" | Sidebar → ⚙️ AI Settings → অন্য model বাছো, বা "Custom model name"-এ সঠিক নাম লেখো |
| "rate limit / quota" | Sidebar → ⚙️ AI Settings → অন্য provider (Groq / Ollama) বাছো, অথবা ১ মিনিট অপেক্ষা করো |
| Ollama: "cannot reach Ollama on this computer" | `OLLAMA_HOST=http://localhost:11434` দিলে Mac-এ Ollama app চালু রাখতে হবে। Cloud ব্যবহার করতে লাইনটা মুছে দাও |
| Port 8501 busy | আগের run বন্ধ করো (■ Stop), বা `streamlit run app.py --server.port 8502` |
| Scanned PDF-এ AI Search কাজ করে না | My Library → **👁️ Read scanned pages with AI Vision** চাপো |
| "cannot read images" | Sidebar → **👁️ Vision AI** → **Gemini** বাছো |
| সব data মুছে নতুন শুরু | app বন্ধ করে `data` folder delete করো |

## 🧰 Technologies
Python · Streamlit · PyMuPDF · ChromaDB · sentence-transformers (`intfloat/multilingual-e5-small`) · SQLite · bcrypt · Google Gemini API (`google-genai`) · Anthropic Claude API (`anthropic`) · Groq API · Ollama API (`httpx`) · gTTS
