"""SentinelNLP — FastAPI entry point."""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.nlp import router

app = FastAPI(title="SentinelNLP", description="Enterprise text analytics API — sentiment, entities, classification, and Claude AI deep analysis.", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router, prefix="/api/v1", tags=["nlp"])

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0", "ai_enabled": bool(os.environ.get("ANTHROPIC_API_KEY"))}

@app.get("/")
def root():
    return {"name": "SentinelNLP", "docs": "/docs"}
