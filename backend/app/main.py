from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Personal AI Dashboard API", version="1.1.0")

# Allow the GitHub Pages dashboard to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://deven3003.github.io",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {
        "service": "personal-ai-dashboard",
        "status": "online",
        "message": "Personal AI backend is running"
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "personal-ai-dashboard"}

@app.post("/ask")
def ask(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return {
        "answer": f"Your Personal AI received: {question}",
        "status": "success",
        "rag_status": "backend connected; real AI/RAG will be enabled next"
    }
