# Personal Dashboard Backend

FastAPI backend for the Personal Dashboard.

## Endpoints

GET /health
POST /ask

This V1 is intentionally a small deployment foundation. Next versions will add:
- RAG ingestion
- embeddings
- vector search
- document management
- persistent conversations
- chat history
- authentication

## Deployment

The repository root is the existing personal-dashboard project.
Upload/merge this `backend/` folder into that repository and deploy it as a Render Web Service using the included render.yaml.

Do not put API keys or passwords in GitHub.
