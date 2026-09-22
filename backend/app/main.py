import json
import os
import uuid
from datetime import datetime, timezone

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel

app = FastAPI(
    title="Personal AI Dashboard API",
    version="3.0.0",
    description="Personal AI backend with Groq and persistent conversations",
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


class ConversationCreate(BaseModel):
    title: str = "New conversation"


def database_url() -> str:
    value = os.getenv("DATABASE_URL")
    if not value:
        raise HTTPException(
            status_code=503,
            detail="DATABASE_URL is not configured on the server.",
        )
    return value


def get_db():
    # Supabase shared session pooler is IPv4-compatible with Render.
    return psycopg.connect(database_url(), sslmode="require")


def init_db():
    url = os.getenv("DATABASE_URL")
    if not url:
        return

    try:
        with psycopg.connect(url, sslmode="require") as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT 'New conversation',
                    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            conn.commit()
    except Exception:
        # Keep the service able to start; endpoints will return a useful
        # configuration error if the database is unavailable.
        pass


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/")
def root():
    return {
        "service": "personal-ai-dashboard",
        "status": "online",
        "features": ["groq-ai", "persistent-conversations"],
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
    }


def get_groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on the server.",
        )
    return Groq(api_key=key)


def call_ai(messages):
    client = get_groq_client()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    system_prompt = """You are Personal AI, the user's private AI assistant inside their Personal Dashboard.

Be helpful, practical, concise, and clear.
Maintain continuity with the conversation history.
You can help with software engineering, career planning, learning, productivity, business planning, and general questions.
Do not invent personal facts or claim access to documents that have not yet been connected.
Persistent conversation history is enabled now. Personal document RAG will be added later."""

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.4,
        max_completion_tokens=1024,
    )
    return completion.choices[0].message.content, model


def read_conversation(conversation_id):
    try:
        with get_db() as conn:
            row = conn.execute(
                """
                SELECT id, title, messages, created_at, updated_at
                FROM conversations
                WHERE id = %s
                """,
                (conversation_id,),
            ).fetchone()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database request failed: {type(exc).__name__}",
        )

    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "id": row[0],
        "title": row[1],
        "messages": row[2],
        "created_at": row[3].isoformat(),
        "updated_at": row[4].isoformat(),
    }


@app.post("/conversations")
def create_conversation(request: ConversationCreate):
    conversation_id = str(uuid.uuid4())
    title = request.title.strip() or "New conversation"

    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, messages)
                VALUES (%s, %s, %s::jsonb)
                """,
                (conversation_id, title, json.dumps([])),
            )
            conn.commit()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database request failed: {type(exc).__name__}",
        )

    return read_conversation(conversation_id)


@app.get("/conversations")
def list_conversations():
    try:
        with get_db() as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at,
                       jsonb_array_length(messages) AS message_count
                FROM conversations
                ORDER BY updated_at DESC
                """
            ).fetchall()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Database request failed: {type(exc).__name__}",
        )

    return {
        "conversations": [
            {
                "id": row[0],
                "title": row[1],
                "created_at": row[2].isoformat(),
                "updated_at": row[3].isoformat(),
                "message_count": row[4],
            }
            for row in rows
        ]
    }


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

    try:
        answer, model = call_ai(messages)

        messages.append({"role": "assistant", "content": answer})

        new_title = conversation["title"]
        if new_title == "New conversation":
            new_title = question[:60]

        with get_db() as conn:
            conn.execute(
                """
                UPDATE conversations
                SET title = %s,
                    messages = %s::jsonb,
                    updated_at = %s
                WHERE id = %s
                """,
                (
                    new_title,
                    json.dumps(messages),
                    datetime.now(timezone.utc),
                    conversation_id,
                ),
            )
            conn.commit()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI provider request failed: {type(exc).__name__}",
        )

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "status": "success",
        "ai_provider": "groq",
        "model": model,
        "message_count": len(messages),
    }


# Keep the old endpoint available for compatibility.
@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    client = get_groq_client()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are Personal AI. Answer clearly and helpfully.",
            },
            {"role": "user", "content": question},
        ],
        temperature=0.4,
        max_completion_tokens=1024,
    )

    return {
        "answer": completion.choices[0].message.content,
        "status": "success",
        "ai_provider": "groq",
        "model": model,
        "rag_status": "not enabled yet",
    }
