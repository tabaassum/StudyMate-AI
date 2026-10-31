"""
rag.py - RAG / VECTOR DATABASE (ChromaDB + sentence-transformers)

How it works:
  1. PDF text is split into small overlapping chunks.
  2. Every chunk is turned into an embedding (a list of numbers that
     represents its MEANING) by a multilingual model (English + Bangla).
  3. Embeddings are stored in ChromaDB (data/chroma_db) with metadata
     (user_id, doc_id, file_name, page).
  4. To search, the question is also embedded and ChromaDB returns the
     chunks whose vectors are closest -> this is the "Retrieval" in RAG.
"""
import re

import chromadb
from chromadb.config import Settings

import config

COLLECTION_NAME = "pdf_chunks"

_model = None
_collection = None


# ---------------------------------------------------------------------------
# Embedding model + vector database
# ---------------------------------------------------------------------------
def get_model():
    """Load the embedding model once (first time it downloads ~470 MB)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
            embedding_function=None,             # we provide embeddings ourselves
        )
    return _collection


def _is_e5() -> bool:
    return "e5" in config.EMBEDDING_MODEL.lower()


def embed_passages(texts):
    if _is_e5():  # E5 models expect these prefixes
        texts = [f"passage: {t}" for t in texts]
    return get_model().encode(texts, normalize_embeddings=True, batch_size=32).tolist()


def embed_query(text):
    if _is_e5():
        text = f"query: {text}"
    return get_model().encode([text], normalize_embeddings=True)[0].tolist()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Split text into ~size character chunks, preferring to cut at sentence ends."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # try to finish at the end of a sentence (English '.', Bangla '।')
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("।"), window.rfind("\n"), window.rfind("? "))
            if cut > size * 0.5:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---------------------------------------------------------------------------
# Indexing / deleting
# ---------------------------------------------------------------------------
def index_document(doc_id, user_id, file_name, pages, progress_cb=None) -> int:
    """
    pages: list of (page_number, text) from pdf_utils.extract_pages
    progress_cb: optional function(fraction 0..1) for a progress bar
    Returns the number of chunks stored.
    """
    ids, docs, metas = [], [], []
    for page_no, text in pages:
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"doc{doc_id}_p{page_no}_c{i}")
            docs.append(chunk)
            metas.append({"user_id": int(user_id), "doc_id": int(doc_id),
                          "file_name": file_name, "page": int(page_no)})
    if not docs:
        return 0

    col = get_collection()
    batch = 64
    for s in range(0, len(docs), batch):
        e = s + batch
        # upsert = insert, or replace if the same chunk id already exists (e.g. a page read again by OCR)
        col.upsert(ids=ids[s:e], documents=docs[s:e], metadatas=metas[s:e],
                   embeddings=embed_passages(docs[s:e]))
        if progress_cb:
            progress_cb(min(1.0, e / len(docs)))
    return len(docs)


def delete_document(doc_id):
    get_collection().delete(where={"doc_id": int(doc_id)})


# ---------------------------------------------------------------------------
# Semantic search (Retrieval)
# ---------------------------------------------------------------------------
def search(query, user_id, doc_ids=None, k=config.TOP_K):
    """
    Return the k most similar chunks:
    [{"text", "doc_id", "file_name", "page", "score"}]  (score 0..1, higher = closer)
    """
    if not query.strip():
        return []
    col = get_collection()
    if col.count() == 0:
        return []

    if doc_ids:
        doc_ids = [int(d) for d in doc_ids]
        doc_filter = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
        where = {"$and": [{"user_id": int(user_id)}, doc_filter]}
    else:
        where = {"user_id": int(user_id)}

    res = col.query(query_embeddings=[embed_query(query)], n_results=k, where=where,
                    include=["documents", "metadatas", "distances"])

    results = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({
            "text": text,
            "doc_id": meta["doc_id"],
            "file_name": meta["file_name"],
            "page": meta["page"],
            "score": round(max(0.0, 1 - dist), 3),
        })
    return results
