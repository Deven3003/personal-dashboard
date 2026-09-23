import json
import os
import re
import uuid
from datetime import datetime, timezone, timedelta
from io import BytesIO

import psycopg
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader
from docx import Document as DocxDocument

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Xenova/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
MAX_RETRIEVAL = 6


def normalize_vector(vector):
    values = [float(x) for x in vector]
    norm = sum(x * x for x in values) ** 0.5
    if norm == 0:
        return values
    return [x / norm for x in values]


def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


app = FastAPI(
    title="Personal AI Dashboard API",
    version="8.0.0",
    description="Personal AI backend with persistent conversations, semantic RAG, source citations, page references, multi-document retrieval, and document-scoped conversations.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str
    use_documents: bool = True
    query_embedding: list[float] | None = None
    document_ids: list[str] | None = None


class ConversationCreate(BaseModel):
    title: str = "New conversation"
    document_ids: list[str] = Field(default_factory=list)


class ConversationScopeUpdate(BaseModel):
    document_ids: list[str] = Field(default_factory=list)


class EmbeddingItem(BaseModel):
    chunk_id: str
    embedding: list[float]


class EmbeddingBatchRequest(BaseModel):
    embeddings: list[EmbeddingItem]


def database_url() -> str:
    value = os.getenv("DATABASE_URL")
    if not value:
        raise HTTPException(status_code=503, detail="DATABASE_URL is not configured on the server.")
    return value


def get_db():
    return psycopg.connect(database_url(), sslmode="require")


def init_db():
    url = os.getenv("DATABASE_URL")
    if not url:
        return
    try:
        with psycopg.connect(url, sslmode="require") as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New conversation',
                    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    size_bytes BIGINT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS document_ids JSONB NOT NULL DEFAULT '[]'::jsonb")
            conn.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding JSONB")
            conn.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT")
            conn.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS page_number INTEGER")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC)")
            conn.commit()
    except Exception:
        pass


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/")
def root():
    return {
        "service": "personal-ai-dashboard",
        "status": "online",
        "version": "8.0.0",
        "features": [
            "groq-ai", "persistent-conversations", "semantic-vector-retrieval",
            "source-citations", "page-references", "multi-document-qa", "document-scoped-conversations", "temporal-reasoning", "numeric-validation"
        ],
    }


@app.get("/health")
def health():
    db_configured = bool(os.getenv("DATABASE_URL"))
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    db_ok = False
    if db_configured:
        try:
            with get_db() as conn:
                conn.execute("SELECT 1")
                db_ok = True
        except Exception:
            db_ok = False
    return {
        "status": "ok" if groq_configured and db_ok else "degraded",
        "service": "personal-ai-dashboard",
        "ai_provider": "groq",
        "ai_configured": groq_configured,
        "database": "postgres",
        "database_configured": db_configured,
        "database_connected": db_ok,
        "document_retrieval": "semantic-vector-client" if db_ok else "waiting_for_database",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": EMBEDDING_DIM,
        "embedding_generation": "browser-side-free",
        "citations": "source-and-page-aware",
        "multi_document": True,
    }


def get_groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="GROQ_API_KEY is not configured on the server.")
    return Groq(api_key=key)


def tokenize(text: str):
    return set(re.findall(r"[a-zA-Z0-9_]{3,}", text.lower()))


def split_text(text: str, chunk_size: int = 1200, overlap: int = 150):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def extract_document_parts(filename: str, content_type: str, raw: bytes):
    lower = filename.lower()
    if lower.endswith(".pdf") or content_type == "application/pdf":
        reader = PdfReader(BytesIO(raw))
        parts = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").replace("\x00", "").strip()
            if text:
                parts.append((page_number, text))
        return parts
    if lower.endswith(".docx") or content_type in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }:
        doc = DocxDocument(BytesIO(raw))
        text = "\n".join(p.text for p in doc.paragraphs).replace("\x00", "").strip()
        return [(None, text)] if text else []
    if lower.endswith(".txt") or content_type.startswith("text/"):
        text = raw.decode("utf-8", errors="replace").replace("\x00", "").strip()
        return [(None, text)] if text else []
    raise HTTPException(status_code=400, detail="Supported files: PDF, DOCX, and TXT.")


def build_chunks(parts):
    output = []
    chunk_index = 0
    for page_number, text in parts:
        for chunk in split_text(text):
            output.append((chunk_index, page_number, chunk))
            chunk_index += 1
    return output


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    raw = file.file.read()
    max_bytes = 10 * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="File is too large. Maximum size is 10 MB.")
    filename = (file.filename or "document").strip()
    try:
        parts = extract_document_parts(filename, file.content_type or "application/octet-stream", raw)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not extract text: {type(exc).__name__}")
    chunks = build_chunks(parts)
    if not chunks:
        raise HTTPException(status_code=400, detail="No usable text could be extracted from this file.")
    document_id = str(uuid.uuid4())
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO documents (id, filename, content_type, size_bytes) VALUES (%s, %s, %s, %s)",
                (document_id, filename, file.content_type or "application/octet-stream", len(raw)),
            )
            for idx, page_number, chunk in chunks:
                conn.execute(
                    """INSERT INTO document_chunks
                       (id, document_id, chunk_index, content, page_number, embedding, embedding_model)
                       VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)""",
                    (str(uuid.uuid4()), document_id, idx, chunk, page_number, None, None),
                )
            conn.commit()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}: {str(exc)[:180]}")
    return {
        "id": document_id,
        "filename": filename,
        "size_bytes": len(raw),
        "chunks": len(chunks),
        "status": "uploaded",
        "retrieval": "semantic-vector-client",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "needs_client_indexing": True,
    }


@app.get("/documents")
def list_documents():
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT d.id, d.filename, d.content_type, d.size_bytes, d.created_at,
                       COUNT(c.id) AS chunk_count,
                       COUNT(c.embedding) AS embedded_count,
                       COUNT(c.page_number) AS page_aware_count
                FROM documents d
                LEFT JOIN document_chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
            """).fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}: {str(exc)[:180]}")
    return {"documents": [
        {
            "id": r[0], "filename": r[1], "content_type": r[2], "size_bytes": r[3],
            "created_at": r[4].isoformat(), "chunks": r[5], "embedded_chunks": r[6],
            "semantic_indexed": r[5] > 0 and r[5] == r[6],
            "page_aware": r[7] > 0,
        } for r in rows
    ]}


@app.get("/documents/index-payload")
def document_index_payload():
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT c.id, c.document_id, d.filename, c.chunk_index, c.page_number, c.content
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NULL
                ORDER BY c.created_at ASC
            """).fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}: {str(exc)[:180]}")
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": EMBEDDING_DIM,
        "chunks": [
            {"id": r[0], "document_id": r[1], "filename": r[2], "chunk_index": r[3], "page_number": r[4], "content": r[5]}
            for r in rows
        ],
    }


@app.post("/documents/embeddings")
def save_document_embeddings(request: EmbeddingBatchRequest):
    if not request.embeddings:
        return {"status": "nothing_to_index", "saved": 0}
    saved = 0
    try:
        with get_db() as conn:
            for item in request.embeddings:
                vector = normalize_vector(item.embedding)
                if len(vector) != EMBEDDING_DIM:
                    raise HTTPException(status_code=400, detail=f"Embedding must have {EMBEDDING_DIM} dimensions")
                cur = conn.execute(
                    "UPDATE document_chunks SET embedding=%s::jsonb, embedding_model=%s WHERE id=%s",
                    (json.dumps(vector), EMBEDDING_MODEL_NAME, item.chunk_id),
                )
                saved += cur.rowcount
            conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}: {str(exc)[:180]}")
    return {"status": "saved", "saved": saved, "embedding_model": EMBEDDING_MODEL_NAME}


@app.post("/documents/reindex")
def reindex_documents():
    try:
        with get_db() as conn:
            row = conn.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NULL").fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}: {str(exc)[:180]}")
    pending = int(row[0])
    return {
        "status": "already_indexed" if pending == 0 else "needs_client_indexing",
        "pending_chunks": pending,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": EMBEDDING_DIM,
    }


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    try:
        with get_db() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            conn.commit()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}: {str(exc)[:180]}")
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "id": document_id}


def relevance_label(score, rank):
    if rank <= 2:
        return "high"
    if rank <= 4:
        return "medium"
    return "related"


def is_date_question(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in (
        "expire", "expires", "expiry", "expiration", "renew", "renewal",
        "valid", "validity", "deadline", "due date", "upcoming", "soon",
        "when does", "when will", "end date", "policy end", "coverage period",
    ))


def temporal_chunk_score(content: str, semantic_score: float, today=None):
    """Score date-bearing chunks for expiry/validity questions.

    Semantic similarity alone can select an older certificate/quote that contains
    a historical policy period. For date questions, explicit end/expiry labels
    and the date's relation to today are stronger signals.
    """
    today = today or datetime.now(timezone(timedelta(hours=5, minutes=30))).date()
    lower = content.lower()
    anchors = (
        "policy end date", "end date", "expiry date", "expiry", "expires",
        "expiration", "valid until", "validity", "policy period",
        "coverage period", "period of insurance", "renewal date", "renewal",
    )
    anchor_hits = sum(1 for a in anchors if a in lower)
    dates = extract_document_dates(content)
    if not anchor_hits or not dates:
        return None

    best = None
    for found in dates:
        value = found["date"]
        if value >= today:
            days = (value - today).days
            # Prefer upcoming dates, with a strong bonus for explicit end/expiry labels.
            proximity = max(0.0, 2.8 - min(days, 365) / 365 * 1.8)
            status_bonus = 3.0
        else:
            days_past = (today - value).days
            proximity = max(0.0, 0.8 - min(days_past, 730) / 730 * 0.6)
            status_bonus = 0.0
        score = float(semantic_score or 0.0) * 0.45 + anchor_hits * 1.25 + status_bonus + proximity
        candidate = (score, found)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best


def retrieve_document_context(question: str, query_vector=None, limit: int = MAX_RETRIEVAL, document_ids=None):
    if not question.strip():
        return []
    selected = [x for x in (document_ids or []) if x]
    rows = []
    try:
        with get_db() as conn:
            if selected:
                rows = conn.execute("""
                    SELECT c.id, c.document_id, d.filename, c.content, c.embedding, c.page_number, c.chunk_index
                    FROM document_chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.embedding IS NOT NULL AND c.document_id = ANY(%s)
                """, (selected,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT c.id, c.document_id, d.filename, c.content, c.embedding, c.page_number, c.chunk_index
                    FROM document_chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.embedding IS NOT NULL
                """).fetchall()
    except Exception:
        rows = []

    if rows and query_vector:
        qv = normalize_vector(query_vector)
        scored = []
        for chunk_id, document_id, filename, content, embedding, page_number, chunk_index in rows:
            try:
                score = cosine_similarity(qv, embedding)
            except Exception:
                continue
            scored.append((score, chunk_id, document_id, filename, content, page_number, chunk_index))

        # Date-sensitive questions get a deterministic retrieval path. This avoids
        # selecting an older historical policy period merely because its wording is
        # semantically similar to the question.
        if is_date_question(question):
            today = datetime.now(timezone(timedelta(hours=5, minutes=30))).date()
            temporal = []
            for score, chunk_id, document_id, filename, content, page_number, chunk_index in scored:
                ranked = temporal_chunk_score(content, score, today=today)
                if ranked is not None:
                    temporal.append((ranked[0], score, chunk_id, document_id, filename, content, page_number, chunk_index, ranked[1]))

            if temporal:
                # Keep the strongest date-bearing chunk from each document first,
                # then allow additional supporting chunks when useful.
                temporal.sort(key=lambda x: x[0], reverse=True)
                results = []
                per_doc = {}
                for temporal_score, semantic_score, chunk_id, document_id, filename, content, page_number, chunk_index, found in temporal:
                    if per_doc.get(document_id, 0) >= 2:
                        continue
                    per_doc[document_id] = per_doc.get(document_id, 0) + 1
                    rank = len(results) + 1
                    results.append({
                        "chunk_id": chunk_id,
                        "document_id": document_id,
                        "filename": filename,
                        "content": content,
                        "score": round(float(semantic_score), 4),
                        "relevance": relevance_label(semantic_score, rank),
                        "page_number": page_number,
                        "chunk_index": chunk_index,
                        "retrieval": "semantic-vector-temporal",
                        "temporal_date": found["raw"],
                    })
                    if len(results) >= limit:
                        break
                if results:
                    return results

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        per_doc = {}
        for score, chunk_id, document_id, filename, content, page_number, chunk_index in scored:
            # Diversify multi-document retrieval while still allowing the strongest source to dominate.
            if per_doc.get(document_id, 0) >= 3:
                continue
            per_doc[document_id] = per_doc.get(document_id, 0) + 1
            rank = len(results) + 1
            results.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "filename": filename,
                "content": content,
                "score": round(float(score), 4),
                "relevance": relevance_label(score, rank),
                "page_number": page_number,
                "chunk_index": chunk_index,
                "retrieval": "semantic-vector",
            })
            if len(results) >= limit:
                break
        return results

    q_tokens = tokenize(question)
    if not q_tokens:
        return []
    try:
        with get_db() as conn:
            if selected:
                legacy_rows = conn.execute("""
                    SELECT c.id, c.document_id, d.filename, c.content, c.page_number, c.chunk_index
                    FROM document_chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.embedding IS NULL AND c.document_id = ANY(%s)
                """, (selected,)).fetchall()
            else:
                legacy_rows = conn.execute("""
                    SELECT c.id, c.document_id, d.filename, c.content, c.page_number, c.chunk_index
                    FROM document_chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.embedding IS NULL
                """).fetchall()
    except Exception:
        return []
    scored = []
    for chunk_id, document_id, filename, content, page_number, chunk_index in legacy_rows:
        overlap = len(q_tokens & tokenize(content))
        if overlap:
            score = overlap / max(1, len(q_tokens))
            scored.append((score, chunk_id, document_id, filename, content, page_number, chunk_index))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "chunk_id": cid, "document_id": did, "filename": f, "content": c,
            "score": round(float(s), 3), "relevance": relevance_label(s, i + 1),
            "page_number": page, "chunk_index": ci, "retrieval": "keyword-fallback"
        }
        for i, (s, cid, did, f, c, page, ci) in enumerate(scored[:limit])
    ]


def source_reference(item, number):
    page = f"page {item['page_number']}" if item.get("page_number") else f"chunk {int(item.get('chunk_index', 0)) + 1}"
    return f"[SOURCE {number}] {item['filename']} — {page} — relevance {item.get('score', 0):.3f}"


MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_date_value(day, month, year):
    try:
        value = datetime(int(year), int(month), int(day)).date()
        return value
    except (TypeError, ValueError):
        return None


def extract_document_dates(text):
    """Extract common human/ISO dates without requiring third-party date parsing."""
    patterns = [
        (re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", re.I), "long"),
        (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*(\d{4})\b", re.I), "long"),
        (re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"), "numeric"),
        (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "iso"),
    ]
    found = []
    seen = set()
    for pattern, kind in patterns:
        for match in pattern.finditer(text):
            if kind == "long":
                day, month_name, year = match.groups()
                month = MONTHS[month_name.lower()]
                value = parse_date_value(day, month, year)
            elif kind == "numeric":
                day, month, year = match.groups()
                value = parse_date_value(day, month, year)
            else:
                year, month, day = match.groups()
                value = parse_date_value(day, month, year)
            if not value or value in seen:
                continue
            seen.add(value)
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            found.append({
                "date": value,
                "raw": match.group(0),
                "context": text[start:end].replace("\n", " "),
            })
    return sorted(found, key=lambda x: x["date"])


def build_reasoning_validation(question, document_context):
    """Create deterministic guardrails for date-sensitive and numeric questions.

    The LLM still writes the answer, but it receives machine-derived facts so it
    cannot casually describe a past expiry as an upcoming one.
    """
    if not document_context:
        return None

    q = question.lower()
    date_intent = any(term in q for term in (
        "expire", "expires", "expiry", "expiration", "renew", "renewal",
        "valid", "validity", "deadline", "due date", "due", "upcoming",
        "soon", "when does", "when will",
    ))
    numeric_intent = any(term in q for term in (
        "amount", "amounts", "premium", "price", "cost", "paid", "pay",
        "sum", "total", "fee", "fees", "compare", "difference", "how much",
        "calculate", "calculation",
    ))
    if not date_intent and not numeric_intent:
        return None

    today = datetime.now(timezone(timedelta(hours=5, minutes=30))).date()
    lines = [f"VALIDATION DATE (India): {today.strftime('%d %B %Y')}"]

    if date_intent:
        expiry_terms = ("expiry", "expires", "expiration", "expire", "expiry date", "end date", "policy end", "valid until", "validity", "policy period", "coverage period", "period of insurance", "renewal", "renew")
        candidates = []
        for item in document_context:
            for found in extract_document_dates(item.get("content", "")):
                context_lower = found["context"].lower()
                if any(term in context_lower for term in expiry_terms):
                    candidates.append((item, found))
        if candidates:
            lines.append("DATE FACTS FROM DOCUMENTS (machine-checked):")
            for item, found in candidates[:12]:
                value = found["date"]
                if value < today:
                    status = f"PAST — { (today - value).days } days before today's date"
                elif value == today:
                    status = "TODAY"
                else:
                    status = f"FUTURE — { (value - today).days } days from today"
                page = f"page {item.get('page_number')}" if item.get("page_number") else f"chunk {int(item.get('chunk_index', 0)) + 1}"
                lines.append(f"- {item['filename']} ({page}): {found['raw']} => {status}")
            lines.append("RULE: For an expiry question, prefer an explicitly labeled Policy End Date / Expiry Date / End Date over a historical date appearing in a certificate, receipt, or older coverage period. If multiple dates exist for the same policy, use the explicitly labeled end date and explain conflicting historical dates. If a document date is PAST, never describe it as a future/upcoming expiry. Say that the date has already passed and clearly distinguish that from any later renewal not present in the documents.")
        else:
            lines.append("No machine-validated expiry/validity date was found near date phrases in the retrieved document context. Do not invent one.")

    if numeric_intent:
        lines.append("NUMERIC RULES: Preserve monetary values exactly as stated in the retrieved sources. Do not invent, round, or silently change amounts. For comparisons, show the source value and label. For arithmetic, verify the calculation before stating a result; if the source does not provide enough inputs, say so.")
        arithmetic = re.compile(r"(₹?\s*[\d,]+(?:\.\d+)?)\s*\+\s*(₹?\s*[\d,]+(?:\.\d+)?)\s*=\s*(₹?\s*[\d,]+(?:\.\d+)?)")
        for item in document_context:
            for match in arithmetic.finditer(item.get("content", "")):
                def amount(raw):
                    return float(raw.replace("₹", "").replace(",", "").strip())
                try:
                    a, b, claimed = map(amount, match.groups())
                    expected = a + b
                    lines.append(f"- Arithmetic check in {item['filename']}: {match.group(0)} => expected total ₹{expected:,.2f}; claimed total ₹{claimed:,.2f}.")
                except ValueError:
                    pass

    return "\n".join(lines)


def call_ai(messages, document_context=None, reasoning_validation=None):
    client = get_groq_client()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    system_prompt = """You are Personal AI, the user's private AI assistant inside their Personal Dashboard.
Be helpful, practical, concise, and clear. Maintain continuity with conversation history.
When document context is provided, answer using it when relevant. Do not invent facts from documents.
If the supplied document context does not contain the answer, say that the available documents do not provide enough information, then answer generally only if useful.
Multiple documents may be supplied. Compare them when the user asks for comparison, differences, common points, or a combined answer.
When document context is supplied, source labels such as [SOURCE 1] are authoritative metadata. You may mention source filenames and page numbers, but NEVER invent a page number or source.
For date-sensitive questions, treat the machine-checked validation facts supplied below as authoritative. Explicitly distinguish a date stated in a document from your calculation relative to today's date.
For numeric questions, preserve source amounts exactly and verify arithmetic before presenting a calculated result.
Prefer a clean structure with headings, bullets, and Markdown tables when comparison or tabular information is useful."""
    if document_context:
        context_text = "\n\n".join(
            f"{source_reference(x, i + 1)}\n{x['content']}" for i, x in enumerate(document_context)
        )
        system_prompt += "\n\nDOCUMENT CONTEXT:\n" + context_text
    if reasoning_validation:
        system_prompt += "\n\nDETERMINISTIC REASONING VALIDATION:\n" + reasoning_validation
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.3,
        max_completion_tokens=1400,
    )
    return completion.choices[0].message.content, model


def read_conversation(conversation_id):
    try:
        with get_db() as conn:
            row = conn.execute("""
                SELECT id, title, messages, created_at, updated_at, COALESCE(document_ids, '[]'::jsonb)
                FROM conversations WHERE id = %s
            """, (conversation_id,)).fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}: {str(exc)[:180]}")
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id": row[0], "title": row[1], "messages": row[2],
        "created_at": row[3].isoformat(), "updated_at": row[4].isoformat(),
        "document_ids": row[5] or [],
    }


@app.post("/conversations")
def create_conversation(request: ConversationCreate):
    conversation_id = str(uuid.uuid4())
    title = request.title.strip() or "New conversation"
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, messages, document_ids) VALUES (%s, %s, %s::jsonb, %s::jsonb)",
                (conversation_id, title, json.dumps([]), json.dumps(request.document_ids)),
            )
            conn.commit()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}: {str(exc)[:180]}")
    return read_conversation(conversation_id)


@app.get("/conversations")
def list_conversations():
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT id, title, created_at, updated_at, jsonb_array_length(messages), COALESCE(document_ids, '[]'::jsonb)
                FROM conversations ORDER BY updated_at DESC
            """).fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}: {str(exc)[:180]}")
    return {"conversations": [
        {"id": r[0], "title": r[1], "created_at": r[2].isoformat(), "updated_at": r[3].isoformat(), "message_count": r[4], "document_ids": r[5] or []}
        for r in rows
    ]}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    return read_conversation(conversation_id)


@app.put("/conversations/{conversation_id}/scope")
def update_conversation_scope(conversation_id: str, request: ConversationScopeUpdate):
    # Empty list means all documents.
    try:
        with get_db() as conn:
            cur = conn.execute(
                "UPDATE conversations SET document_ids=%s::jsonb, updated_at=%s WHERE id=%s",
                (json.dumps(request.document_ids), datetime.now(timezone.utc), conversation_id),
            )
            conn.commit()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}: {str(exc)[:180]}")
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return read_conversation(conversation_id)


@app.post("/conversations/{conversation_id}/ask")
def conversation_ask(conversation_id: str, request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    conversation = read_conversation(conversation_id)
    messages = conversation["messages"] or []
    messages.append({"role": "user", "content": question})
    scope = request.document_ids if request.document_ids is not None else conversation.get("document_ids", [])
    context = retrieve_document_context(question, request.query_embedding, document_ids=scope) if request.use_documents else []
    reasoning_validation = build_reasoning_validation(question, context)
    try:
        answer, model = call_ai(messages, context, reasoning_validation)
        messages.append({"role": "assistant", "content": answer})
        new_title = conversation["title"] if conversation["title"] != "New conversation" else question[:60]
        with get_db() as conn:
            conn.execute("""
                UPDATE conversations SET title=%s, messages=%s::jsonb, updated_at=%s WHERE id=%s
            """, (new_title, json.dumps(messages), datetime.now(timezone.utc), conversation_id))
            conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"AI provider request failed: {type(exc).__name__}")

    sources = []
    for i, x in enumerate(context, start=1):
        sources.append({
            "source_number": i,
            "filename": x["filename"],
            "document_id": x["document_id"],
            "page_number": x.get("page_number"),
            "chunk_index": x.get("chunk_index"),
            "score": x.get("score"),
            "relevance": x.get("relevance"),
            "retrieval": x.get("retrieval"),
        })
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "status": "success",
        "ai_provider": "groq",
        "model": model,
        "message_count": len(messages),
        "document_sources": [x["filename"] for x in context],
        "sources": sources,
        "retrieval_mode": context[0].get("retrieval") if context else ("disabled" if not request.use_documents else "no_match"),
        "retrieval_scores": [x.get("score") for x in context],
        "document_scope": scope,
        "reasoning_validation": reasoning_validation is not None,
    }


@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    context = retrieve_document_context(question, request.query_embedding, document_ids=request.document_ids) if request.use_documents else []
    reasoning_validation = build_reasoning_validation(question, context)
    answer, model = call_ai([{"role": "user", "content": question}], context, reasoning_validation)
    return {
        "answer": answer,
        "status": "success",
        "ai_provider": "groq",
        "model": model,
        "rag_status": "document retrieval enabled",
        "sources": [
            {
                "source_number": i,
                "filename": x["filename"],
                "document_id": x["document_id"],
                "page_number": x.get("page_number"),
                "chunk_index": x.get("chunk_index"),
                "score": x.get("score"),
                "relevance": x.get("relevance"),
            }
            for i, x in enumerate(context, start=1)
        ],
        "reasoning_validation": reasoning_validation is not None,
    }
