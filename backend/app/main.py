import os

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI(
    title="Personal AI Dashboard API",
    version="2.0.1",
    description="Personal AI backend with Groq LLM integration",
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


@app.options("/ask")
def ask_options():
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


@app.get("/")
def root():
    return {
        "service": "personal-ai-dashboard",
        "status": "online",
        "ai": "groq",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "personal-ai-dashboard",
        "ai_provider": "groq",
        "ai_configured": bool(os.getenv("GROQ_API_KEY")),
    }


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured on the server.",
        )
    return Groq(api_key=api_key)


@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    client = get_client()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    system_prompt = """You are Personal AI, the user's private AI assistant inside their Personal Dashboard.

Be helpful, practical, concise, and clear.
You can help with software engineering, career planning, learning, productivity, business planning, and general questions.
Do not pretend to know personal facts that have not been provided in the conversation or connected knowledge sources.
For this first version, answer using your general knowledge. Personal documents, persistent memory, and RAG will be connected in later versions."""

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.4,
            max_completion_tokens=1024,
        )

        answer = completion.choices[0].message.content

        return {
            "answer": answer,
            "status": "success",
            "ai_provider": "groq",
            "model": model,
            "rag_status": "not enabled yet",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AI provider request failed: {type(exc).__name__}",
        )
