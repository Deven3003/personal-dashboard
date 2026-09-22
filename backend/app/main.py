import json
import os
import re
import uuid
from datetime import datetime, timezone
from io import BytesIO

import psycopg
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document as DocxDocument

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Xenova/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384


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
    version="6.0.0",
    description="Personal AI backend with Groq, persistent conversations, multi-document semantic vector retrieval using browser-side embeddings, and chat history",
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


class ConversationCreate(BaseModel):
    title: str = "New conversation"


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
            conn.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding JSONB")
            conn.execute("ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT")
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
        "features": ["groq-ai", "persistent-conversations", "multi-document-retrieval"],
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
        "embedding_generation": "browser-side-free"
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


def extract_text(filename: str, content_type: str, raw: bytes) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf") or content_type == "application/pdf":
        reader = PdfReader(BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if lower.endswith(".docx") or content_type in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }:
        doc = DocxDocument(BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    if lower.endswith(".txt") or content_type.startswith("text/"):
        return raw.decode("utf-8", errors="replace")
    raise HTTPException(status_code=400, detail="Supported files: PDF, DOCX, and TXT.")


@app.post("/documents")
def upload_document(file: UploadFile = File(...)):
    raw = file.file.read()
    max_bytes = 10 * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="File is too large. Maximum size is 10 MB.")
    filename = (file.filename or "document").strip()
    try:
        text = extract_text(filename, file.content_type or "application/octet-stream", raw)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not extract text: {type(exc).__name__}")
    text = text.strip()
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="No usable text could be extracted from this file.")
    chunks = split_text(text)
    document_id = str(uuid.uuid4())

    # Embeddings are generated in the user's browser, not on Render.
    # This keeps the free 512 MB Render instance lightweight.

    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO documents (id, filename, content_type, size_bytes) VALUES (%s, %s, %s, %s)",
                (document_id, filename, file.content_type or "application/octet-stream", len(raw)),
            )
            for idx, chunk in enumerate(chunks):
                conn.execute(
                    """INSERT INTO document_chunks
                       (id, document_id, chunk_index, content, embedding, embedding_model)
                       VALUES (%s, %s, %s, %s, %s::jsonb, %s)""",
                    (
                        str(uuid.uuid4()), document_id, idx, chunk,
                        None, None,
                    ),
                )
            conn.commit()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}")
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
                       COUNT(c.embedding) AS embedded_count
                FROM documents d
                LEFT JOIN document_chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
            """).fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}")
    return {"documents": [
        {
            "id": r[0], "filename": r[1], "content_type": r[2], "size_bytes": r[3],
            "created_at": r[4].isoformat(), "chunks": r[5], "embedded_chunks": r[6],
            "semantic_indexed": r[5] > 0 and r[5] == r[6],
        } for r in rows
    ]}


@app.get("/documents/index-payload")
def document_index_payload():
    """Return unembedded chunks so the browser can create vectors for free."""
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT id, document_id, content
                FROM document_chunks
                WHERE embedding IS NULL
                ORDER BY created_at ASC
            """).fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}")
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimensions": EMBEDDING_DIM,
        "chunks": [{"id": r[0], "document_id": r[1], "content": r[2]} for r in rows],
    }


class EmbeddingItem(BaseModel):
    chunk_id: str
    embedding: list[float]


class EmbeddingBatchRequest(BaseModel):
    embeddings: list[EmbeddingItem]


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
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}")
    return {"status": "saved", "saved": saved, "embedding_model": EMBEDDING_MODEL_NAME}


@app.post("/documents/reindex")
def reindex_documents():
    """Compatibility endpoint: browser must perform the actual embedding work."""
    try:
        with get_db() as conn:
            row = conn.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NULL").fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}")
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
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}")
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "id": document_id}


def retrieve_document_context(question: str, query_vector=None, limit: int = 5):
    """Retrieve using a browser-generated embedding, with keyword fallback."""
    if not question.strip():
        return []
    rows = []
    if query_vector:
        try:
            query_vector = normalize_vector(query_vector)
            with get_db() as conn:
                rows = conn.execute("""
                    SELECT d.filename, c.content, c.embedding
                    FROM document_chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.embedding IS NOT NULL
                """).fetchall()
        except Exception:
            rows = []
    if rows and query_vector:
        scored = []
        for filename, content, embedding in rows:
            try:
                score = cosine_similarity(query_vector, embedding)
            except Exception:
                continue
            scored.append((score, filename, content))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"filename": f, "content": c, "score": round(float(s), 4), "retrieval": "semantic-vector"}
            for s, f, c in scored[:limit]
        ]

    q_tokens = tokenize(question)
    if not q_tokens:
        return []
    try:
        with get_db() as conn:
            legacy_rows = conn.execute("""
                SELECT d.filename, c.content
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NULL
                ORDER BY c.created_at DESC
            """).fetchall()
    except Exception:
        return []
    scored = []
    for filename, content in legacy_rows:
        tokens = tokenize(content)
        overlap = len(q_tokens & tokens)
        if overlap:
            score = overlap / max(1, len(q_tokens))
            scored.append((score, filename, content))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"filename": f, "content": c, "score": round(s, 3), "retrieval": "keyword-fallback"}
        for s, f, c in scored[:limit]
    ]


def call_ai(messages, document_context=None):
    client = get_groq_client()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    system_prompt = """You are Personal AI, the user's private AI assistant inside their Personal Dashboard.
Be helpful, practical, concise, and clear. Maintain continuity with conversation history.
When document context is provided, answer using it when relevant. Do not invent facts from documents.
If the supplied document context does not contain the answer, say that the available documents do not provide enough information, then answer generally only if useful.
When using document context, mention the source filename naturally in the answer.
Persistent conversation history and semantic vector document retrieval are enabled. When document context includes retrieval metadata, prefer the highest-scoring relevant passages."""
    if document_context:
        context_text = "\n\n".join(
            f"SOURCE: {x['filename']}\n{x['content']}" for x in document_context
        )
        system_prompt += "\n\nDOCUMENT CONTEXT:\n" + context_text
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.3,
        max_completion_tokens=1200,
    )
    return completion.choices[0].message.content, model


def read_conversation(conversation_id):
    try:
        with get_db() as conn:
            row = conn.execute("""
                SELECT id, title, messages, created_at, updated_at
                FROM conversations WHERE id = %s
            """, (conversation_id,)).fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}")
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"id": row[0], "title": row[1], "messages": row[2], "created_at": row[3].isoformat(), "updated_at": row[4].isoformat()}


@app.post("/conversations")
def create_conversation(request: ConversationCreate):
    conversation_id = str(uuid.uuid4())
    title = request.title.strip() or "New conversation"
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO conversations (id, title, messages) VALUES (%s, %s, %s::jsonb)", (conversation_id, title, json.dumps([])))
            conn.commit()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}")
    return read_conversation(conversation_id)


@app.get("/conversations")
def list_conversations():
    try:
        with get_db() as conn:
            rows = conn.execute("""
                SELECT id, title, created_at, updated_at, jsonb_array_length(messages)
                FROM conversations ORDER BY updated_at DESC
            """).fetchall()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database request failed: {type(exc).__name__}")
    return {"conversations": [
        {"id": r[0], "title": r[1], "created_at": r[2].isoformat(), "updated_at": r[3].isoformat(), "message_count": r[4]}
        for r in rows
    ]}


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    return read_conversation(conversation_id)


@app.post("/conversations/{conversation_id}/ask")
def conversation_ask(conversation_id: str, request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    conversation = read_conversation(conversation_id)
    messages = conversation["messages"] or []
    messages.append({"role": "user", "content": question})
    context = retrieve_document_context(question, request.query_embedding) if request.use_documents else []
    try:
        answer, model = call_ai(messages, context)
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
    return {
        "conversation_id": conversation_id, "answer": answer, "status": "success",
        "ai_provider": "groq", "model": model, "message_count": len(messages),
        "document_sources": [x["filename"] for x in context],
        "retrieval_mode": context[0].get("retrieval") if context else ("disabled" if not request.use_documents else "no_match"),
        "retrieval_scores": [x.get("score") for x in context],
    }


@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer, model = call_ai([{"role": "user", "content": question}])
    return {"answer": answer, "status": "success", "ai_provider": "groq", "model": model, "rag_status": "document retrieval enabled"}
