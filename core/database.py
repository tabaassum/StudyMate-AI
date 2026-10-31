"""
database.py - DATABASE CONNECTION (SQLite)

Stores all structured data of the app:
  users, documents (upload history), reading progress, bookmarks, notes,
  chat history, search history, summaries, translations and quiz results.

SQLite needs no server - the whole database is one file: data/studymate.db
(You can open it with "DB Browser for SQLite" or PyCharm's Database tool.)
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import bcrypt

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    email         TEXT,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    file_name   TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    page_count  INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    uploaded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reading_progress (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    last_page  INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, doc_id)
);

CREATE TABLE IF NOT EXISTS bookmarks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page       INTEGER NOT NULL,
    label      TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (user_id, doc_id, page)
);

CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page       INTEGER NOT NULL,
    content    TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (user_id, doc_id, page)
);

CREATE TABLE IF NOT EXISTS chat_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question   TEXT NOT NULL,
    answer     TEXT NOT NULL,
    sources    TEXT,            -- JSON list of {doc_id, file_name, page}
    provider   TEXT,
    feedback   INTEGER,         -- 1 = like, -1 = dislike, NULL = none
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query       TEXT NOT NULL,
    search_type TEXT NOT NULL,  -- 'ai' or 'keyword'
    results     INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    scope      TEXT NOT NULL,   -- 'all' or '3-7'
    style      TEXT NOT NULL,
    language   TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS translations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    direction   TEXT NOT NULL,  -- 'en-bn' or 'bn-en'
    source_text TEXT NOT NULL,
    translated  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ocr_pages (      -- text read from scanned pages by AI Vision
    doc_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page       INTEGER NOT NULL,
    text       TEXT NOT NULL,
    provider   TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (doc_id, page)
);

CREATE TABLE IF NOT EXISTS vision_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source     TEXT NOT NULL,   -- 'image' or 'pdf: name p.3'
    task       TEXT NOT NULL,
    answer     TEXT NOT NULL,
    provider   TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS quiz_results (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    doc_id     INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    doc_name   TEXT,
    score      INTEGER NOT NULL,
    total      INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def get_conn():
    """Open a connection, commit on success, always close."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _rows(rows):
    return [dict(r) for r in rows]


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def create_user(username: str, email: str, password: str):
    """Returns (True, user_id) or (False, error message)."""
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    try:
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (username.strip(), email.strip(), pw_hash, _now()),
            )
            return True, cur.lastrowid
    except sqlite3.IntegrityError:
        return False, "This username is already taken."


def verify_user(username: str, password: str):
    """Returns the user dict if username + password are correct, else None."""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
    if row and bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
        user = dict(row)
        user.pop("password_hash", None)
        return user
    return None


# ---------------------------------------------------------------------------
# Documents (upload history)
# ---------------------------------------------------------------------------
def add_document(user_id, file_name, file_path, page_count):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (user_id, file_name, file_path, page_count, uploaded_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, file_name, file_path, page_count, _now()),
        )
        return cur.lastrowid


def set_chunk_count(doc_id, chunk_count):
    with get_conn() as conn:
        conn.execute("UPDATE documents SET chunk_count = ? WHERE id = ?", (chunk_count, doc_id))


def get_documents(user_id):
    with get_conn() as conn:
        return _rows(conn.execute(
            "SELECT * FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC, id DESC", (user_id,)
        ).fetchall())


def get_document(doc_id, user_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id)).fetchone()
        return dict(row) if row else None


def delete_document(doc_id, user_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (doc_id, user_id))


def get_recent_documents(user_id, limit=5):
    """Documents ordered by when they were last read (falls back to upload date)."""
    with get_conn() as conn:
        return _rows(conn.execute(
            """SELECT d.*, rp.last_page, COALESCE(rp.updated_at, d.uploaded_at) AS last_used
               FROM documents d
               LEFT JOIN reading_progress rp ON rp.doc_id = d.id AND rp.user_id = d.user_id
               WHERE d.user_id = ?
               ORDER BY last_used DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall())


# ---------------------------------------------------------------------------
# Reading progress
# ---------------------------------------------------------------------------
def set_progress(user_id, doc_id, page):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO reading_progress (user_id, doc_id, last_page, updated_at) VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, doc_id) DO UPDATE SET last_page = excluded.last_page,
                                                          updated_at = excluded.updated_at""",
            (user_id, doc_id, page, _now()),
        )


def get_progress(user_id, doc_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT last_page FROM reading_progress WHERE user_id = ? AND doc_id = ?", (user_id, doc_id)
        ).fetchone()
        return row["last_page"] if row else 1


# ---------------------------------------------------------------------------
# Bookmarks
# ---------------------------------------------------------------------------
def add_bookmark(user_id, doc_id, page, label=""):
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO bookmarks (user_id, doc_id, page, label, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, doc_id, page, label, _now()),
        )


def remove_bookmark(user_id, doc_id, page):
    with get_conn() as conn:
        conn.execute("DELETE FROM bookmarks WHERE user_id = ? AND doc_id = ? AND page = ?", (user_id, doc_id, page))


def get_bookmarks(user_id, doc_id):
    with get_conn() as conn:
        return _rows(conn.execute(
            "SELECT * FROM bookmarks WHERE user_id = ? AND doc_id = ? ORDER BY page", (user_id, doc_id)
        ).fetchall())


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
def save_note(user_id, doc_id, page, content):
    with get_conn() as conn:
        if content.strip():
            conn.execute(
                """INSERT INTO notes (user_id, doc_id, page, content, updated_at) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(user_id, doc_id, page) DO UPDATE SET content = excluded.content,
                                                                    updated_at = excluded.updated_at""",
                (user_id, doc_id, page, content.strip(), _now()),
            )
        else:
            conn.execute("DELETE FROM notes WHERE user_id = ? AND doc_id = ? AND page = ?", (user_id, doc_id, page))


def get_note(user_id, doc_id, page):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT content FROM notes WHERE user_id = ? AND doc_id = ? AND page = ?", (user_id, doc_id, page)
        ).fetchone()
        return row["content"] if row else ""


def get_notes(user_id, doc_id):
    with get_conn() as conn:
        return _rows(conn.execute(
            "SELECT * FROM notes WHERE user_id = ? AND doc_id = ? ORDER BY page", (user_id, doc_id)
        ).fetchall())


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
def add_chat(user_id, question, answer, sources, provider):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chat_history (user_id, question, answer, sources, provider, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, question, answer, json.dumps(sources, ensure_ascii=False), provider, _now()),
        )
        return cur.lastrowid


def get_chats(user_id, limit=200):
    with get_conn() as conn:
        rows = _rows(conn.execute(
            "SELECT * FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)
        ).fetchall())
    for r in rows:
        r["sources"] = json.loads(r["sources"]) if r.get("sources") else []
    return rows


def set_feedback(chat_id, user_id, value):
    with get_conn() as conn:
        conn.execute("UPDATE chat_history SET feedback = ? WHERE id = ? AND user_id = ?", (value, chat_id, user_id))


def delete_chat(chat_id, user_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_history WHERE id = ? AND user_id = ?", (chat_id, user_id))


def clear_chats(user_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))


# ---------------------------------------------------------------------------
# Search history
# ---------------------------------------------------------------------------
def add_search(user_id, query, search_type, results):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO search_history (user_id, query, search_type, results, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, query, search_type, results, _now()),
        )


def get_searches(user_id, limit=100):
    with get_conn() as conn:
        return _rows(conn.execute(
            "SELECT * FROM search_history WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)
        ).fetchall())


# ---------------------------------------------------------------------------
# Summaries (cache, so the same summary is not generated twice)
# ---------------------------------------------------------------------------
def get_summary(user_id, doc_id, scope, style, language):
    with get_conn() as conn:
        row = conn.execute(
            """SELECT * FROM summaries WHERE user_id = ? AND doc_id = ? AND scope = ? AND style = ? AND language = ?
               ORDER BY id DESC LIMIT 1""",
            (user_id, doc_id, scope, style, language),
        ).fetchone()
        return dict(row) if row else None


def save_summary(user_id, doc_id, scope, style, language, content):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO summaries (user_id, doc_id, scope, style, language, content, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, doc_id, scope, style, language, content, _now()),
        )


# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------
def add_translation(user_id, direction, source_text, translated):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO translations (user_id, direction, source_text, translated, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, direction, source_text, translated, _now()),
        )


def get_translations(user_id, limit=50):
    with get_conn() as conn:
        return _rows(conn.execute(
            "SELECT * FROM translations WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)
        ).fetchall())


# ---------------------------------------------------------------------------
# AI Vision: OCR text of scanned pages + vision history
# ---------------------------------------------------------------------------
def save_ocr_text(doc_id, page, text, provider=""):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO ocr_pages (doc_id, page, text, provider, created_at) VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(doc_id, page) DO UPDATE SET text = excluded.text, provider = excluded.provider,
                                                      created_at = excluded.created_at""",
            (doc_id, page, text, provider, _now()),
        )


def get_ocr_text(doc_id, page):
    with get_conn() as conn:
        row = conn.execute("SELECT text FROM ocr_pages WHERE doc_id = ? AND page = ?", (doc_id, page)).fetchone()
        return row["text"] if row else ""


def get_ocr_pages(doc_id):
    """{page: text} for every page of this document that AI Vision has read."""
    with get_conn() as conn:
        rows = conn.execute("SELECT page, text FROM ocr_pages WHERE doc_id = ?", (doc_id,)).fetchall()
        return {r["page"]: r["text"] for r in rows}


def add_vision(user_id, source, task, answer, provider):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO vision_history (user_id, source, task, answer, provider, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, source, task, answer, provider, _now()),
        )


def get_vision_history(user_id, limit=100):
    with get_conn() as conn:
        return _rows(conn.execute(
            "SELECT * FROM vision_history WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)
        ).fetchall())


# ---------------------------------------------------------------------------
# Quiz results
# ---------------------------------------------------------------------------
def add_quiz_result(user_id, doc_id, doc_name, score, total):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO quiz_results (user_id, doc_id, doc_name, score, total, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, doc_id, doc_name, score, total, _now()),
        )


def get_quiz_results(user_id):
    with get_conn() as conn:
        return _rows(conn.execute(
            "SELECT * FROM quiz_results WHERE user_id = ? ORDER BY id DESC", (user_id,)
        ).fetchall())


# ---------------------------------------------------------------------------
# Dashboard statistics
# ---------------------------------------------------------------------------
def get_stats(user_id):
    with get_conn() as conn:
        one = lambda sql: conn.execute(sql, (user_id,)).fetchone()[0]  # noqa: E731
        stats = {
            "documents": one("SELECT COUNT(*) FROM documents WHERE user_id = ?"),
            "pages": one("SELECT COALESCE(SUM(page_count), 0) FROM documents WHERE user_id = ?"),
            "questions": one("SELECT COUNT(*) FROM chat_history WHERE user_id = ?"),
            "searches": one("SELECT COUNT(*) FROM search_history WHERE user_id = ?"),
            "translations": one("SELECT COUNT(*) FROM translations WHERE user_id = ?"),
            "quizzes": one("SELECT COUNT(*) FROM quiz_results WHERE user_id = ?"),
            "vision": one("SELECT COUNT(*) FROM vision_history WHERE user_id = ?"),
            "likes": one("SELECT COUNT(*) FROM chat_history WHERE user_id = ? AND feedback = 1"),
            "dislikes": one("SELECT COUNT(*) FROM chat_history WHERE user_id = ? AND feedback = -1"),
        }
        row = conn.execute(
            "SELECT SUM(score), SUM(total) FROM quiz_results WHERE user_id = ?", (user_id,)
        ).fetchone()
        stats["quiz_avg"] = round(100 * row[0] / row[1]) if row and row[1] else None
    return stats
