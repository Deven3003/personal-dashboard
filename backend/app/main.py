from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Personal AI Dashboard API", version="1.0.0")

class QuestionRequest(BaseModel):
    question: str

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
        "rag_status": "backend foundation ready; RAG will be enabled next"
    }
